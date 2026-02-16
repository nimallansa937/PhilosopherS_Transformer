           "routing\_logits": routing\_logits,  
            "routing\_decision": \["SELF", "ORACLE", "HYBRID"\]\[  
                routing\_logits.argmax(dim=-1).item()\],  
            "error\_logits": error\_logits,  
            "error\_type": \[  
                "NONE", "FACTUAL\_GAP", "REASONING\_ERROR",  
                "FORMALIZATION\_ERROR", "SCOPE\_EXCEEDED"  
            \]\[error\_logits.argmax(dim=-1).item()\],  
            "features": features,  \# Cache for feedback loop  
        }

class FeedbackBuffer:  
    """Stores interaction outcomes for online meta-learner training.  
      
    The core of the feedback-forward mechanism. Every interaction   
    produces a training signal that improves future routing.  
    """  
      
    def \_\_init\_\_(self, max\_size: int \= 10000):  
        self.buffer \= deque(maxlen=max\_size)  
        self.batch\_size \= 32  
      
    def record(self,   
               features: torch.Tensor,  
               predicted\_confidence: float,  
               predicted\_routing: str,  
               actual\_outcome: dict):  
        """Record an interaction outcome.  
          
        actual\_outcome contains:  
          \- "oracle\_agreed": bool (did oracle confirm small model?)  
          \- "z3\_verified": bool | None (did Z3 confirm formal claims?)  
          \- "user\_accepted": bool | None (did user accept answer?)  
          \- "correction\_magnitude": float (how much oracle changed answer)  
        """  
          
        \# Compute ground truth confidence from outcome signals  
        true\_confidence \= self.\_compute\_true\_confidence(actual\_outcome)  
          
        \# Compute ground truth routing  
        true\_routing \= self.\_compute\_true\_routing(  
            actual\_outcome, true\_confidence)  
          
        \# Compute error type  
        true\_error \= self.\_compute\_error\_type(actual\_outcome)  
          
        self.buffer.append({  
            "features": features.detach(),  
            "true\_confidence": true\_confidence,  
            "true\_routing": true\_routing,  
            "true\_error": true\_error,  
            "predicted\_confidence": predicted\_confidence,  
            "predicted\_routing": predicted\_routing,  
            "timestamp": torch.tensor(len(self.buffer)),  
        })  
      
    def \_compute\_true\_confidence(self, outcome: dict) \-\> float:  
        """Derive true confidence from multiple outcome signals.  
          
        Weighted combination:  
        \- Z3 verification: strongest signal (binary, reliable)  
        \- Oracle agreement: strong signal (but oracle can be wrong too)  
        \- User acceptance: weak signal (users accept wrong answers sometimes)  
        """  
        signals \= \[\]  
        weights \= \[\]  
          
        if outcome.get("z3\_verified") is not None:  
            signals.append(1.0 if outcome\["z3\_verified"\] else 0.0)  
            weights.append(3.0)  \# Highest weight — Z3 is ground truth  
          
        if outcome.get("oracle\_agreed") is not None:  
            signals.append(1.0 if outcome\["oracle\_agreed"\] else 0.0)  
            weights.append(2.0)  
          
        if outcome.get("correction\_magnitude") is not None:  
            \# Low correction \= high confidence was justified  
            signals.append(1.0 \- min(outcome\["correction\_magnitude"\], 1.0))  
            weights.append(1.5)  
          
        if outcome.get("user\_accepted") is not None:  
            signals.append(1.0 if outcome\["user\_accepted"\] else 0.3)  
            weights.append(0.5)  \# Weakest signal  
          
        if not signals:  
            return 0.5  \# No information  
          
        return sum(s \* w for s, w in zip(signals, weights)) / sum(weights)  
      
    def \_compute\_true\_routing(self, outcome: dict,   
                               true\_conf: float) \-\> int:  
        """Derive what routing SHOULD have been.  
          
        0=SELF, 1=ORACLE, 2=HYBRID  
        """  
        if true\_conf \>= 0.85:  
            return 0  \# Should have handled it alone  
        elif true\_conf \< 0.4:  
            return 1  \# Should have deferred entirely  
        else:  
            return 2  \# Hybrid was appropriate  
      
    def \_compute\_error\_type(self, outcome: dict) \-\> int:  
        """Classify what went wrong (if anything).  
          
        0=NONE, 1=FACTUAL\_GAP, 2=REASONING\_ERROR,   
        3=FORMALIZATION\_ERROR, 4=SCOPE\_EXCEEDED  
        """  
        if outcome.get("z3\_verified") is False:  
            return 3  \# Formalization error — Z3 rejected it  
          
        if outcome.get("oracle\_agreed") is False:  
            mag \= outcome.get("correction\_magnitude", 0.5)  
            if mag \> 0.7:  
                return 1  \# Major factual gap  
            else:  
                return 2  \# Reasoning error  
          
        if outcome.get("correction\_magnitude", 0\) \> 0.8:  
            return 4  \# Scope exceeded — question was out of domain  
          
        return 0  \# No error  
      
    def sample\_batch(self) \-\> Optional\[dict\]:  
        """Sample a training batch from the buffer."""  
        if len(self.buffer) \< self.batch\_size:  
            return None  
          
        \# Prioritized sampling: recent interactions weighted higher  
        \# because the small model may have been updated  
        indices \= list(range(len(self.buffer)))  
        weights \= \[1.0 \+ i / len(self.buffer) for i in indices\]  
        total\_w \= sum(weights)  
        probs \= \[w / total\_w for w in weights\]  
          
        import random  
        sampled\_idx \= random.choices(indices, weights=probs,   
                                      k=self.batch\_size)  
          
        batch \= \[self.buffer\[i\] for i in sampled\_idx\]  
          
        return {  
            "features": torch.stack(\[b\["features"\].squeeze(0)   
                                     for b in batch\]),  
            "true\_confidence": torch.tensor(  
                \[b\["true\_confidence"\] for b in batch\]),  
            "true\_routing": torch.tensor(  
                \[b\["true\_routing"\] for b in batch\], dtype=torch.long),  
            "true\_error": torch.tensor(  
                \[b\["true\_error"\] for b in batch\], dtype=torch.long),  
        }

class MetaLearnerTrainer:  
    """Online trainer for the meta-learner.  
      
    Runs in the background after each oracle consultation.  
    Each oracle interaction produces a ground-truth label,  
    which trains the meta-learner to be better calibrated  
    for future queries.  
    """  
      
    def \_\_init\_\_(self, meta\_learner: MetaLearner,   
                 lr: float \= 1e-4):  
        self.meta \= meta\_learner  
        self.buffer \= FeedbackBuffer()  
        self.optimizer \= torch.optim.AdamW(  
            meta\_learner.parameters(), lr=lr)  
          
        \# Loss functions  
        self.confidence\_loss \= nn.MSELoss()  
        self.routing\_loss \= nn.CrossEntropyLoss()  
        self.error\_loss \= nn.CrossEntropyLoss()  
          
        \# Loss weights — confidence matters most  
        self.w\_conf \= 2.0  
        self.w\_route \= 1.5  
        self.w\_error \= 1.0  
          
        \# Training stats  
        self.update\_count \= 0  
        self.loss\_history \= deque(maxlen=1000)  
      
    def record\_outcome(self, features: torch.Tensor,  
                       predicted\_confidence: float,  
                       predicted\_routing: str,  
                       actual\_outcome: dict):  
        """Record an interaction and maybe train."""  
          
        self.buffer.record(features, predicted\_confidence,  
                          predicted\_routing, actual\_outcome)  
          
        \# Train every 8 new interactions  
        if len(self.buffer) \>= 32 and len(self.buffer) % 8 \== 0:  
            self.\_train\_step()  
      
    def \_train\_step(self):  
        """One gradient step on buffered feedback."""  
          
        batch \= self.buffer.sample\_batch()  
        if batch is None:  
            return  
          
        self.meta.train()  
        self.optimizer.zero\_grad()  
          
        \# Forward through meta-learner heads directly from cached features  
        features \= batch\["features"\]  
          
        conf\_pred \= self.meta.confidence\_head(features).squeeze(-1)  
        route\_pred \= self.meta.routing\_head(features)  
        error\_pred \= self.meta.error\_head(features)  
          
        \# Compute losses  
        loss\_conf \= self.confidence\_loss(  
            conf\_pred, batch\["true\_confidence"\])  
        loss\_route \= self.routing\_loss(  
            route\_pred, batch\["true\_routing"\])  
        loss\_error \= self.error\_loss(  
            error\_pred, batch\["true\_error"\])  
          
        total\_loss \= (self.w\_conf \* loss\_conf \+   
                      self.w\_route \* loss\_route \+   
                      self.w\_error \* loss\_error)  
          
        total\_loss.backward()  
        torch.nn.utils.clip\_grad\_norm\_(self.meta.parameters(), 1.0)  
        self.optimizer.step()  
          
        self.update\_count \+= 1  
        self.loss\_history.append(total\_loss.item())  
          
        self.meta.eval()  
          
        if self.update\_count % 50 \== 0:  
            avg\_loss \= sum(self.loss\_history) / len(self.loss\_history)  
            print(f"  \[Meta-learner\] Update {self.update\_count}, "  
                  f"avg loss: {avg\_loss:.4f}")  
      
    def save(self, path: str):  
        """Save meta-learner state \+ buffer."""  
        torch.save({  
            "model\_state": self.meta.state\_dict(),  
            "optimizer\_state": self.optimizer.state\_dict(),  
            "update\_count": self.update\_count,  
            "buffer\_size": len(self.buffer),  
        }, path)  
      
    def load(self, path: str):  
        """Restore meta-learner state."""  
        checkpoint \= torch.load(path)  
        self.meta.load\_state\_dict(checkpoint\["model\_state"\])  
        self.optimizer.load\_state\_dict(checkpoint\["optimizer\_state"\])  
        self.update\_count \= checkpoint\["update\_count"\]

Now the key piece — how the signal extractor hooks into the small model during generation:

\# \~/inference/signal\_extractor.py  
"""  
Extracts meta-learner input signals from the small model  
during generation. Hooks into model internals without   
modifying the model itself.  
"""

import torch  
import numpy as np  
import re  
from typing import List  
from meta\_learner import ModelSignals

HEDGE\_WORDS \= {  
    "perhaps", "possibly", "might", "may", "could",   
    "uncertain", "unclear", "debatable", "arguably",  
    "not sure", "i believe", "it seems", "roughly",  
    "approximately", "likely", "unlikely"  
}

class SignalExtractor:  
    """Hooks into model forward pass to extract uncertainty signals."""  
      
    def \_\_init\_\_(self, model, tokenizer):  
        self.model \= model  
        self.tokenizer \= tokenizer  
        self.hidden\_states \= \[\]  
        self.attention\_maps \= \[\]  
        self.\_register\_hooks()  
      
    def \_register\_hooks(self):  
        """Register forward hooks on the last transformer layer."""  
          
        \# Find last layer — works for most HuggingFace models  
        layers \= None  
        if hasattr(self.model, 'model'):  
            if hasattr(self.model.model, 'layers'):  
                layers \= self.model.model.layers  
          
        if layers is not None:  
            last\_layer \= layers\[-1\]  
            last\_layer.register\_forward\_hook(self.\_capture\_hidden)  
              
            \# Hook attention if accessible  
            if hasattr(last\_layer, 'self\_attn'):  
                last\_layer.self\_attn.register\_forward\_hook(  
                    self.\_capture\_attention)  
      
    def \_capture\_hidden(self, module, input, output):  
        """Capture hidden states from last layer."""  
        if isinstance(output, tuple):  
            self.hidden\_states.append(output\[0\].detach())  
        else:  
            self.hidden\_states.append(output.detach())  
      
    def \_capture\_attention(self, module, input, output):  
        """Capture attention weights."""  
        if isinstance(output, tuple) and len(output) \> 1:  
            attn\_weights \= output\[1\]  
            if attn\_weights is not None:  
                self.attention\_maps.append(attn\_weights.detach())  
      
    def clear(self):  
        """Clear captured states for next generation."""  
        self.hidden\_states \= \[\]  
        self.attention\_maps \= \[\]  
      
    def extract\_signals(self,   
                        input\_ids: torch.Tensor,  
                        generated\_ids: torch.Tensor,  
                        generated\_text: str) \-\> ModelSignals:  
        """Extract all signals after a generation is complete.  
          
        Args:  
            input\_ids: The prompt token IDs  
            generated\_ids: The full sequence (prompt \+ generated)  
            generated\_text: Decoded generated text  
        """  
          
        \# \--- Hidden state statistics \---  
        if self.hidden\_states:  
            \# Use only the generated portion's hidden states  
            last\_hidden \= self.hidden\_states\[-1\]  
            prompt\_len \= input\_ids.shape\[-1\]  
            gen\_hidden \= last\_hidden\[:, prompt\_len:, :\]  
              
            h\_mean \= gen\_hidden.mean(dim=1).squeeze(0)  \# \[hidden\_dim\]  
            h\_std \= gen\_hidden.std(dim=1).squeeze(0)  
        else:  
            hidden\_dim \= 4096  \# Default for 8B models  
            h\_mean \= torch.zeros(hidden\_dim)  
            h\_std \= torch.ones(hidden\_dim)  
          
        \# \--- Token-level entropy \---  
        token\_entropies \= self.\_compute\_token\_entropies(  
            input\_ids, generated\_ids)  
          
        \# \--- Attention entropy \---  
        attn\_entropy \= self.\_compute\_attention\_entropy()  
          
        \# \--- Textual signals \---  
        hedge\_count \= sum(  
            1 for hw in HEDGE\_WORDS   
            if hw in generated\_text.lower()  
        )  
          
        rep\_rate \= self.\_compute\_repetition\_rate(generated\_text)  
          
        \# \--- Query similarity to training distribution \---  
        \# Proxy: use mean hidden state norm   
        \# (in-distribution inputs produce more typical activations)  
        query\_sim \= self.\_compute\_distribution\_similarity(h\_mean)  
          
        \# \--- Topic embedding \---  
        \# Use the hidden state of the first generated token as   
        \# a rough topic embedding  
        if self.hidden\_states and last\_hidden.shape\[1\] \> prompt\_len:  
            topic\_emb \= last\_hidden\[0, prompt\_len, :256\]  
        else:  
            topic\_emb \= torch.zeros(256)  
          
        return ModelSignals(  
            hidden\_state\_mean=h\_mean,  
            hidden\_state\_std=h\_std,  
            token\_entropies=token\_entropies,  
            attention\_entropy=attn\_entropy,  
            topic\_embedding=topic\_emb,  
            hedge\_word\_count=hedge\_count,  
            repetition\_rate=rep\_rate,  
            query\_similarity=query\_sim,  
        )  
      
    def \_compute\_token\_entropies(self,   
                                  input\_ids: torch.Tensor,  
                                  full\_ids: torch.Tensor) \-\> List\[float\]:  
        """Compute per-token generation entropy.  
          
        High entropy tokens \= model is uncertain about that token.  
        Sustained high entropy \= model is in uncertain territory.  
        """  
        entropies \= \[\]  
          
        with torch.no\_grad():  
            outputs \= self.model(full\_ids, output\_hidden\_states=False)  
            logits \= outputs.logits  \# \[1, seq\_len, vocab\_size\]  
          
        prompt\_len \= input\_ids.shape\[-1\]  
        gen\_logits \= logits\[0, prompt\_len-1:-1, :\]  \# Logits for generated tokens  
          
        for i in range(gen\_logits.shape\[0\]):  
            probs \= torch.softmax(gen\_logits\[i\], dim=-1)  
            entropy \= \-torch.sum(probs \* torch.log(probs \+ 1e-10)).item()  
            entropies.append(entropy)  
          
        return entropies  
      
    def \_compute\_attention\_entropy(self) \-\> float:  
        """Compute average attention entropy across heads.  
          
        Dispersed attention (high entropy) → model is uncertain,  
        looking everywhere for relevant context.  
        Focused attention (low entropy) → model knows exactly   
        what to attend to.  
        """  
        if not self.attention\_maps:  
            return 5.0  \# Default moderate entropy  
          
        last\_attn \= self.attention\_maps\[-1\]  \# \[batch, heads, seq, seq\]  
        \# Average across heads and positions  
        attn\_probs \= last\_attn\[0\].mean(dim=0)  \# \[seq, seq\]  
          
        \# Entropy of each position's attention distribution  
        eps \= 1e-10  
        entropies \= \-torch.sum(  
            attn\_probs \* torch.log(attn\_probs \+ eps), dim=-1)  
          
        return entropies.mean().item()  
      
    def \_compute\_repetition\_rate(self, text: str) \-\> float:  
        """Detect repeated phrases (sign of confabulation).  
          
        Models that don't know the answer tend to repeat   
        themselves or generate circular text.  
        """  
        words \= text.lower().split()  
        if len(words) \< 20:  
            return 0.0  
          
        \# Check 4-gram repetitions  
        ngrams \= \[tuple(words\[i:i+4\]) for i in range(len(words)-3)\]  
        unique\_ratio \= len(set(ngrams)) / max(len(ngrams), 1\)  
          
        return 1.0 \- unique\_ratio  \# 0 \= no repetition, 1 \= all repeated  
      
    def \_compute\_distribution\_similarity(self,   
                                          h\_mean: torch.Tensor) \-\> float:  
        """Estimate how close this input is to training distribution.  
          
        Simple proxy: L2 norm of mean hidden state. In-distribution   
        inputs produce activations with typical norms; OOD inputs   
        produce unusually large or small norms.  
          
        In production: replace with a proper OOD detector   
        (Mahalanobis distance from training distribution statistics).  
        """  
        norm \= h\_mean.norm().item()  
        \# Typical norm for 8B model hidden states is \~30-60  
        \# Normalize to \[0, 1\] range  
        typical\_norm \= 45.0  
        deviation \= abs(norm \- typical\_norm) / typical\_norm  
        return max(0.0, 1.0 \- deviation)

Now the updated cascade engine that wires everything together:

\# \~/inference/cascade\_engine\_v2.py  
"""  
Cascade inference engine with meta-learner feedback loop.  
Replaces the text-based \[CONFIDENCE\] parsing with   
learned confidence from model internals.  
"""

import torch  
import json  
import os  
from typing import Optional  
from transformers import AutoModelForCausalLM, AutoTokenizer

from meta\_learner import MetaLearner, MetaLearnerTrainer  
from signal\_extractor import SignalExtractor  
from oracle import OracleClient, OracleConfig

class DescartesEngineV2:  
    """Production engine with meta-learner routing."""  
      
    def \_\_init\_\_(self,   
                 model\_path: str,  
                 meta\_learner\_path: Optional\[str\] \= None,  
                 oracle\_config: OracleConfig \= None):  
          
        \# Load small model  
        print("Loading Descartes model...")  
        self.tokenizer \= AutoTokenizer.from\_pretrained(  
            model\_path, trust\_remote\_code=True)  
        self.model \= AutoModelForCausalLM.from\_pretrained(  
            model\_path, torch\_dtype=torch.bfloat16,  
            device\_map="auto", trust\_remote\_code=True)  
        self.model.eval()  
          
        \# Signal extractor (hooks into model)  
        self.extractor \= SignalExtractor(self.model, self.tokenizer)  
          
        \# Meta-learner  
        self.meta \= MetaLearner(  
            hidden\_dim=self.model.config.hidden\_size,  
            feature\_dim=256  
        )  
        if meta\_learner\_path and os.path.exists(meta\_learner\_path):  
            checkpoint \= torch.load(meta\_learner\_path)  
            self.meta.load\_state\_dict(checkpoint\["model\_state"\])  
            print(f"Loaded meta-learner ({checkpoint\['update\_count'\]} updates)")  
        self.meta.eval()  
          
        \# Online trainer (feedback loop)  
        self.trainer \= MetaLearnerTrainer(self.meta)  
          
        \# Oracle  
        self.oracle \= OracleClient(oracle\_config or OracleConfig())  
          
        \# Stats  
        self.total\_queries \= 0  
        self.self\_handled \= 0  
        self.oracle\_handled \= 0  
        self.hybrid\_handled \= 0  
          
        print("Engine ready (v2 with meta-learner).")  
      
    def generate\_with\_signals(self, prompt: str,   
                               max\_new\_tokens: int \= 2048,  
                               temperature: float \= 0.3):  
        """Generate response and extract internal signals."""  
          
        self.extractor.clear()  
          
        full\_prompt \= (  
            f"\<|system|\>\\nYou are a philosophical reasoning assistant "  
            f"specializing in Cartesian philosophy.\\n"  
            f"\<|user|\>\\n{prompt}\\n\<|assistant|\>\\n"  
        )  
          
        inputs \= self.tokenizer(  
            full\_prompt, return\_tensors="pt",  
            truncation=True, max\_length=8192  
        ).to(self.model.device)  
          
        with torch.no\_grad():  
            outputs \= self.model.generate(  
                \*\*inputs,  
                max\_new\_tokens=max\_new\_tokens,  
                temperature=temperature,  
                do\_sample=True,  
                top\_p=0.9,  
                pad\_token\_id=self.tokenizer.eos\_token\_id,  
                output\_hidden\_states=False,  
            )  
          
        response\_text \= self.tokenizer.decode(  
            outputs\[0\]\[inputs\["input\_ids"\].shape\[1\]:\],  
            skip\_special\_tokens=True  
        ).strip()  
          
        \# Extract signals from generation  
        signals \= self.extractor.extract\_signals(  
            inputs\["input\_ids"\], outputs\[0:1\], response\_text)  
          
        return response\_text, signals  
      
    def run(self, query: str) \-\> dict:  
        """Full cascade pipeline with meta-learner routing."""  
          
        self.total\_queries \+= 1  
          
        \# Step 1: Generate from small model \+ extract signals  
        response, signals \= self.generate\_with\_signals(query)  
          
        \# Step 2: Meta-learner decides confidence and routing  
        with torch.no\_grad():  
            meta\_output \= self.meta(signals)  
          
        confidence \= meta\_output\["confidence"\].item()  
        routing \= meta\_output\["routing\_decision"\]  
        error\_type \= meta\_output\["error\_type"\]  
        cached\_features \= meta\_output\["features"\]  
          
        result \= {  
            "query": query,  
            "initial\_response": response,  
            "meta\_confidence": confidence,  
            "meta\_routing": routing,  
            "meta\_error\_type": error\_type,  
            "oracle\_used": False,  
            "oracle\_response": None,  
            "final\_response": response,  
            "final\_confidence": confidence,  
        }  
          
        \# Step 3: Route based on meta-learner decision  
        if routing \== "SELF":  
            self.self\_handled \+= 1  
            result\["final\_response"\] \= response  
            result\["final\_confidence"\] \= confidence  
              
        elif routing \== "ORACLE":  
            self.oracle\_handled \+= 1  
              
            \# Construct oracle query based on error type  
            oracle\_query \= self.\_construct\_oracle\_query(  
                query, response, error\_type)  
              
            oracle\_response \= self.oracle.query(  
                oracle\_query, context=response)  
              
            result\["oracle\_used"\] \= True  
            result\["oracle\_response"\] \= oracle\_response  
              
            \# Integration pass  
            integrated, \_ \= self.generate\_with\_signals(  
                f"You answered a question but need to integrate "  
                f"additional knowledge.\\n\\n"  
                f"QUESTION: {query}\\n\\n"  
                f"YOUR ANSWER:\\n{response}\\n\\n"  
                f"ADDITIONAL KNOWLEDGE:\\n{oracle\_response}\\n\\n"  
                f"Produce your final integrated answer."  
            )  
            result\["final\_response"\] \= integrated  
              
            \# Feedback: compare initial vs oracle  
            self.\_record\_feedback(  
                cached\_features, confidence, routing,  
                response, oracle\_response  
            )  
              
        elif routing \== "HYBRID":  
            self.hybrid\_handled \+= 1  
              
            \# Small model handles formal aspects, oracle handles knowledge  
            oracle\_query \= self.\_construct\_oracle\_query(  
                query, response, error\_type)  
              
            oracle\_response \= self.oracle.query(  
                oracle\_query, context=response)  
              
            result\["oracle\_used"\] \= True  
            result\["oracle\_response"\] \= oracle\_response  
              
            integrated, \_ \= self.generate\_with\_signals(  
                f"Integrate your formal analysis with additional "  
                f"philosophical knowledge.\\n\\n"  
                f"QUESTION: {query}\\n\\n"  
                f"YOUR FORMAL ANALYSIS:\\n{response}\\n\\n"  
                f"PHILOSOPHICAL CONTEXT:\\n{oracle\_response}\\n\\n"  
                f"Produce a complete answer combining both."  
            )  
            result\["final\_response"\] \= integrated  
              
            self.\_record\_feedback(  
                cached\_features, confidence, routing,  
                response, oracle\_response  
            )  
          
        return result  
      
    def \_construct\_oracle\_query(self, original\_query: str,  
                                 small\_response: str,  
                                 error\_type: str) \-\> str:  
        """Construct oracle query based on predicted error type.  
          
        The meta-learner tells us WHAT KIND of knowledge gap   
        exists, so we can ask the oracle the right question.  
        """  
          
        if error\_type \== "FACTUAL\_GAP":  
            return (  
                f"A Descartes specialist needs factual philosophical "  
                f"knowledge to answer this question:\\n\\n"  
                f"{original\_query}\\n\\n"  
                f"They've provided this partial analysis:\\n"  
                f"{small\_response\[:500\]}\\n\\n"  
                f"What factual information are they missing?"  
            )  
          
        elif error\_type \== "SCOPE\_EXCEEDED":  
            return (  
                f"This question goes beyond Cartesian philosophy "  
                f"into broader territory. Please provide the "  
                f"relevant context:\\n\\n{original\_query}"  
            )  
          
        elif error\_type \== "REASONING\_ERROR":  
            return (  
                f"A specialist provided this analysis but may "  
                f"have reasoning errors:\\n\\n"  
                f"Question: {original\_query}\\n"  
                f"Analysis: {small\_response\[:500\]}\\n\\n"  
                f"Please check the reasoning and provide corrections."  
            )  
          
        else:  
            return original\_query  
      
    def \_record\_feedback(self, features, predicted\_conf,   
                          predicted\_routing, small\_response,  
                          oracle\_response):  
        """Record outcome for meta-learner training.  
          
        This is the FEEDBACK part of feedback-forward.  
        """  
          
        \# Compute agreement between small model and oracle  
        \# Simple heuristic: cosine similarity of response embeddings  
        \# In production: use a proper semantic similarity model  
        agreement \= self.\_compute\_agreement(  
            small\_response, oracle\_response)  
          
        \# Compute correction magnitude  
        correction\_mag \= 1.0 \- agreement  
          
        outcome \= {  
            "oracle\_agreed": agreement \> 0.7,  
            "correction\_magnitude": correction\_mag,  
            "z3\_verified": None,  \# Set later if Z3 was used  
            "user\_accepted": None,  \# Set later from user feedback  
        }  
          
        self.trainer.record\_outcome(  
            features, predicted\_conf,  
            predicted\_routing, outcome  
        )  
      
    def \_compute\_agreement(self, text\_a: str, text\_b: str) \-\> float:  
        """Quick semantic agreement score between two responses.  
          
        Simple version: Jaccard similarity on content words.  
        Production version: use a sentence embedding model.  
        """  
        words\_a \= set(text\_a.lower().split()) \- {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}  
        words\_b \= set(text\_b.lower().split()) \- {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}  
          
        if not words\_a or not words\_b:  
            return 0.5  
          
        intersection \= words\_a & words\_b  
        union \= words\_a | words\_b  
        return len(intersection) / len(union)  
      
    def record\_user\_feedback(self, accepted: bool):  
        """Call this when user gives explicit feedback.  
          
        Updates the most recent buffer entry with user signal.  
        """  
        if self.trainer.buffer.buffer:  
            last \= self.trainer.buffer.buffer\[-1\]  
            \# Recompute true confidence with user signal  
            \# This is an approximation — in production you'd  
            \# re-derive the full outcome  
            if accepted:  
                last\["true\_confidence"\] \= min(  
                    last\["true\_confidence"\] \+ 0.1, 1.0)  
            else:  
                last\["true\_confidence"\] \= max(  
                    last\["true\_confidence"\] \- 0.2, 0.0)  
      
    def save\_state(self, path: str):  
        """Save meta-learner state for persistence across sessions."""  
        self.trainer.save(path)  
        print(f"Meta-learner saved ({self.trainer.update\_count} updates, "  
              f"buffer size: {len(self.trainer.buffer.buffer)})")  
      
    def get\_stats(self) \-\> dict:  
        return {  
            "total\_queries": self.total\_queries,  
            "self\_handled": self.self\_handled,  
            "oracle\_handled": self.oracle\_handled,  
            "hybrid\_handled": self.hybrid\_handled,  
            "self\_rate": self.self\_handled / max(self.total\_queries, 1),  
            "meta\_learner\_updates": self.trainer.update\_count,  
            "oracle\_stats": self.oracle.get\_stats(),  
            "avg\_meta\_loss": (  
                sum(self.trainer.loss\_history) /   
                max(len(self.trainer.loss\_history), 1\)  
            ) if self.trainer.loss\_history else None,  
        }

Now the critical question — how do you **bootstrap** the meta-learner before it has any feedback data?

\# \~/training/bootstrap\_meta\_learner.py  
"""  
Phase 9 revised: Bootstrap the meta-learner with synthetic   
feedback data before deploying to production.

Method:  
1\. Run the small model on 500-1000 held-out Descartes questions  
2\. Simultaneously run the oracle on the same questions  
3\. Compare answers to generate ground-truth labels  
4\. Pre-train the meta-learner on these labels  
5\. Deploy with a warm-started meta-learner that improves online

This replaces the temperature-scaling calibrator from the   
previous pipeline — the meta-learner is strictly more powerful.  
"""

import torch  
import json  
from pathlib import Path  
from meta\_learner import MetaLearner, MetaLearnerTrainer, ModelSignals  
from signal\_extractor import SignalExtractor  
from oracle import OracleClient, OracleConfig  
from transformers import AutoModelForCausalLM, AutoTokenizer

def bootstrap(model\_path: str,   
              questions\_path: str,  
              output\_path: str,  
              oracle\_config: OracleConfig \= None,  
              max\_questions: int \= 500):  
    """Bootstrap meta-learner from small-model vs oracle comparison."""  
      
    print("Loading model...")  
    tokenizer \= AutoTokenizer.from\_pretrained(  
        model\_path, trust\_remote\_code=True)  
    model \= AutoModelForCausalLM.from\_pretrained(  
        model\_path, torch\_dtype=torch.bfloat16,  
        device\_map="auto", trust\_remote\_code=True)  
    model.eval()  
      
    extractor \= SignalExtractor(model, tokenizer)  
    oracle \= OracleClient(oracle\_config or OracleConfig())  
      
    meta \= MetaLearner(hidden\_dim=model.config.hidden\_size)  
    trainer \= MetaLearnerTrainer(meta, lr=5e-4)  \# Higher LR for bootstrap  
      
    \# Load questions  
    with open(questions\_path) as f:  
        questions \= \[json.loads(line)\["question"\]   
                     for line in f\]\[:max\_questions\]  
      
    print(f"Bootstrapping on {len(questions)} questions...")  
      
    for i, question in enumerate(questions):  
        \# Generate from small model  
        extractor.clear()  
          
        prompt \= f"\<|system|\>\\nCartesian philosophy assistant.\\n\<|user|\>\\n{question}\\n\<|assistant|\>\\n"  
        inputs \= tokenizer(prompt, return\_tensors="pt",  
                          truncation=True, max\_length=4096  
                          ).to(model.device)  
          
        with torch.no\_grad():  
            outputs \= model.generate(  
                \*\*inputs, max\_new\_tokens=1024,  
                temperature=0.3, do\_sample=True,  
                pad\_token\_id=tokenizer.eos\_token\_id)  
          
        response \= tokenizer.decode(  
            outputs\[0\]\[inputs\["input\_ids"\].shape\[1\]:\],  
            skip\_special\_tokens=True).strip()  
          
        signals \= extractor.extract\_signals(  
            inputs\["input\_ids"\], outputs\[0:1\], response)  
          
        \# Get oracle answer  
        oracle\_response \= oracle.query(question)  
          
        \# Compute agreement  
        words\_s \= set(response.lower().split())  
        words\_o \= set(oracle\_response.lower().split())  
        stop \= {"the","a","an","is","are","was","were","in","on","at","to","for","of","and"}  
        words\_s \-= stop  
        words\_o \-= stop  
        agreement \= len(words\_s & words\_o) / max(len(words\_s | words\_o), 1\)  
          
        \# Record as training data  
        with torch.no\_grad():  
            meta\_out \= meta(signals)  
          
        outcome \= {  
            "oracle\_agreed": agreement \> 0.7,  
            "correction\_magnitude": 1.0 \- agreement,  
            "z3\_verified": None,  
            "user\_accepted": None,  
        }  
          
        trainer.record\_outcome(  
            meta\_out\["features"\],  
            meta\_out\["confidence"\].item(),  
            meta\_out\["routing\_decision"\],  
            outcome  
        )  
          
        if (i \+ 1\) % 50 \== 0:  
            stats \= {  
                "questions": i \+ 1,  
                "meta\_updates": trainer.update\_count,  
                "oracle\_cost": oracle.total\_cost,  
            }  
            if trainer.loss\_history:  
                stats\["avg\_loss"\] \= sum(trainer.loss\_history) / len(trainer.loss\_history)  
            print(f"  \[{i+1}/{len(questions)}\] {json.dumps(stats)}")  
      
    \# Save bootstrapped meta-learner  
    trainer.save(output\_path)  
      
    print(f"\\nBootstrap complete.")  
    print(f"  Meta-learner updates: {trainer.update\_count}")  
    print(f"  Oracle cost: ${oracle.total\_cost:.2f}")  
    print(f"  Saved to: {output\_path}")

\# Usage:  
\# bootstrap(  
\#     model\_path="\~/models/descartes-8b-cascade",  
\#     questions\_path="\~/training/eval/bootstrap\_questions.jsonl",  
\#     output\_path="\~/models/meta\_learner\_bootstrapped.pt",  
\#     max\_questions=500  
\# )

\# Cost estimate:  
\# 500 oracle calls × \~500 tokens each × $0.0002/1K tokens   
\# ≈ $0.05 total oracle cost  
\# \+ \~2-4 hours GPU time for small model generation  
\# ≈ $2 GPU cost  
\# Total bootstrap cost: \~$2-$3

Should I add this as a formal Phase 9 revision to the Descartes pipeline document, or do you want to iterate on the design first?

