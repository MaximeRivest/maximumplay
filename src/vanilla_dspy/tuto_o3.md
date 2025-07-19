## 0  Prerequisites  (install & configure)

```bash
pip install -U dspy-ai litellm
export OPENAI_API_KEY="sk‑..."
```

```python
import dspy, os

# pick any chat‑capable backend liteLLM supports
dspy.configure(
    lm      = dspy.LM("openai/gpt-4o-mini"),
    adapter = dspy.ChatAdapter()      # default for comparison
)

print("DSPy version:", dspy.__version__)
```

Expected:

```
DSPy version: 2.x.x
```

<!-- 👉 your output here -->

---

## 1  Anatomy of a **Signature**

```python
class MiniSig(dspy.Signature):
    instructions = "You are a neutral assistant."
    question: str = dspy.InputField()
    answer  : str = dspy.OutputField()
```

* `instructions` → system prompt (trainable)
* `question`/`answer` → schema, not trainable

---

## 2  First call with **Predict**

```python
ask = dspy.Predict(MiniSig)

res = ask(question="What is the capital of Spain?")
print(res.answer)
```

<!-- 👉 your output here -->

---

## 3  Peek at the **formatted prompt**

DSPy doesn’t expose it by default, but you can reconstruct it:

```python
lm      = dspy.settings.lm
adapter = dspy.settings.adapter
inputs  = {"question": "What is the capital of Spain?"}
prompt  = adapter.format(MiniSig, demos=[], inputs=inputs)

# pretty‑print
from pprint import pprint
pprint(prompt)
```

<!-- 👉 your output here -->

---

## 4  Parsing—see the raw JSON → typed output

```python
raw = lm(messages=prompt).completion
parsed = adapter.parse(MiniSig, raw)
print("Parsed dict:", parsed)          # should map back to answer:str
```

<!-- 👉 your output here -->

---

## 5  Working with **demos**

### 5.1  Hard‑coding

```python
demo = dspy.Example(
    question="Who lives in a pineapple under the sea?",
    answer   ="SpongeBob SquarePants!"
).with_inputs("question")

ask.demos = [demo]

_ = ask(question="Finish the phrase: To be or not to be …")
```

Inspect the prompt again:

```python
prompt2 = adapter.format(MiniSig, demos=ask.demos,
                         inputs={"question":"Finish the phrase …"})
pprint(prompt2)
```

<!-- 👉 show the extra few‑shot turn -->

### 5.2  Passing demos **at call time**

```python
demo2 = dspy.Example(
    question="Yarr! Where be the treasure?",
    answer="Arr, on Skull Isle!"
).with_inputs("question")

_ = ask(question="Speak like a pirate!", demos=[demo2])
```

### 5.3  Letting an **optimizer** choose demos

```python
from dspy.optimizers.bootstrap_few_shot import BootstrapFewShot

trainset = [
    demo,
    demo2,
    dspy.Example(question="Hola!", answer="¡Hola!").with_inputs("question"),
]

opt     = BootstrapFewShot(num_demos=2, max_rounds=1)
ask_tuned = opt.compile(student=ask, trainset=trainset,
                        metric=lambda ex, pred: 1.0)   # dummy metric
print("Demos chosen by optimizer:")
for ex in ask_tuned.demos: print("  •", ex.question)
```

<!-- 👉 output shows two selected demos -->

---

## 6  Writing a **custom adapter** (ultra‑minimal)

```python
from dspy.adapters.base   import Adapter
from dspy.adapters.utils  import split_message_content_for_custom_types

class MinimalAdapter(Adapter):
    def format(self, sig, demos, inputs):
        msgs = [{"role":"system", "content": sig.instructions}]
        for d in demos:
            msgs.append({"role":"user",      "content": d.question})
            msgs.append({"role":"assistant", "content": d.answer})
        msgs.append({"role":"user", "content": inputs["question"]})
        return split_message_content_for_custom_types(msgs)

    def parse(self, sig, completion):
        return {"answer": completion.strip()}

dspy.configure(adapter=MinimalAdapter())
```

Run again to verify:

```python
mini = dspy.Predict(MiniSig)
print(mini(question="Say hi!").answer)
```

<!-- 👉 your output here -->

Inspect prompt shape:

```python
pprint(dspy.settings.adapter.format(MiniSig, demos=[], inputs={"question":"Say hi!"}))
```

<!-- 👉 notice no brackets, no schema headers -->

---

## 7  Optimizing the **instruction** string only

```python
from dspy.optimizers.base import BaseOptimizer

class TinyCOPRO(BaseOptimizer):
    def propose(self, instr):
        prm = (f"Rewrite the persona so the assistant speaks pirate.\n"
               f"Current: {instr}\nRewrite:")
        outs = self.lm(text=prm, n=3).completions
        return [o.strip() for o in outs]

    def compile(self, student, *, trainset, metric, **kw):
        best, best_score = student, -1
        for cand in self.propose(student.signature.instructions):
            prog = student.clone()
            prog.signature.instructions = cand
            score = self.evaluate(prog, trainset, metric)
            if score > best_score:
                best, best_score = prog, score
        return best

pirate_metric = lambda ex, pred: int("arr" in pred.answer.lower())
train = [dspy.Example(question="Greet me", answer="Arr!").with_inputs("question")]

optimizer = TinyCOPRO()
pirate_predict = optimizer.compile(student=mini, trainset=train, metric=pirate_metric)

print("New instruction:", pirate_predict.signature.instructions)
print(pirate_predict(question="Hello sailor!").answer)
```

<!-- 👉 verify pirate tone -->

---

## 8  Recap

* **Signature** → schema + *trainable* instruction
* **Adapter** → formatting & parsing, swappable
* **Demos** → few‑shot examples owned by each `Predict` instance
* **Optimizers** → mutate instruction and/or demos; never touch live user input
* **Everything inspectable** → you can always print the prompt, demos, token usage

Feel free to extend this notebook—add history, custom metrics, or chain multiple modules—and paste live outputs where the `👉` markers are.
