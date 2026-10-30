"""Bottom-up attention (BUTD) Faster R-CNN — Visual Genome, 1600 objects / 400 attributes.

Vendored from huggingface/transformers v4.30.0 examples/research_projects/lxmert
(``modeling_frcnn.py``, ``processing_image.py``; Apache-2.0, headers kept). Port of
airsplay/py-bottom-up-attention (Anderson et al., CVPR 2018). ``utils.py`` here is a
minimal replacement: only ``Config``; the original download helpers are left out.
Weights (``unc-nlp/frcnn-vg-finetuned``) are loaded locally by ``scripts/extract_visual.py``.
"""
