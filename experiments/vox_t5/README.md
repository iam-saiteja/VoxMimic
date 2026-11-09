# VOX layer on T5-small (retired)

The first attempt at local rewriting: a frozen T5-small with a trainable linear layer that adds a style vector to the encoder's hidden states, trained on synthetic pairs made by swapping casual words for formal ones.

Kept here for reference. It no longer runs against the current package; check out commit `b71d472` to try it.

## Why it was retired

Run on the same paragraph with a style vector, and with an all-zero one:

```
input:      It is important to note that regular exercise can significantly improve
            overall well-being. Furthermore, individuals who utilize a consistent
            routine often experience enhanced productivity.

with style: that regular exercise can significantly improve overall well-being.
no style:   that regular exercise can significantly improve overall well-being.
```

The outputs are identical, and neither is a rewrite. T5-small was trained for translation and summarising, so without a task prefix it mostly truncates its input. A 0.01-scaled offset to the hidden states can't teach it a skill it doesn't have, and a larger one breaks the output, which is why alpha kept coming down.

## What survived

The training-pair idea did. Pairing someone's real sentences with a formal version of them turned out to be the right idea; it just needed a model that can already paraphrase. `voxmimic/rewrite.py` generates those pairs with a real LLM and uses them as few-shot examples.

`migrate_profile.py` is here too, since it handled an older format of this profile. The current `voxmimic/storage.py` migrates the old numpy format on its own.
