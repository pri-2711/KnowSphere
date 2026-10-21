# Qwen2.5-VL evaluation for document extraction

**Evaluation date:** 2026-10-20  
**Status:** desk evaluation only. No model weights were downloaded, no inference was run, and no hosted API was called.

## What Qwen2.5-VL could add

Qwen2.5-VL is a vision-language model family with 3B, 7B, and 72B checkpoints. Its official release describes reading text, charts, graphics, and page layouts. Official examples show passing page images to the model through Transformers; Alibaba Cloud Model Studio also offers hosted Qwen vision endpoints. See the [Qwen release announcement](https://qwen.ai/blog?from=research.research-list&id=c5e7415d9a9e89adc18c59d9e466e5a1a459b8f4), the [3B model card and Transformers example](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), and [Model Studio image parsing options](https://help.aliyun.com/zh/model-studio/workflow/image-extraction-node).

For this project, a vision model might describe a diagram, read labels embedded in a figure, or help interpret a table as a visual grid. Its response is generated interpretation: it can omit details or invent relationships. It must be stored separately from the source text and reviewed against the page image.

## Approaches

| Approach | Strengths | Limits and risks | Suitability here |
|---|---|---|---|
| PyMuPDF + Tesseract (current) | Already installed; fast for selectable text; local processing; no per-page API charge; current TXT workflow works. | Tesseract is text recognition, not diagram understanding. PyMuPDF `sort=True` does not guarantee correct multi-column order or table structure. Sparse, legitimate text can trigger OCR. | Keep as the default first pass. Fix evidence-backed extraction bookkeeping before adding a model. |
| Qwen2.5-VL on every page | Sees page layout and visual content. | Unnecessary cost/latency for text-rich pages; risks sending whole books to a service; model can hallucinate; local memory and runtime can be substantial. | Not recommended. |
| Hybrid: existing extraction, then selected page images | Preserves a cheap baseline and targets only pages where layout or visual reasoning is valuable. | Requires page selection, image handling, an explicit privacy/cost decision for hosted inference, and review of model output. | Recommended experiment path after approval; do not integrate yet. |

## Local hardware and hosted API considerations

The inspected machine has about **15.8 GiB system RAM**. Its installed PyTorch build reports **CPU-only** (`2.14.0+cpu`, CUDA unavailable), and `nvidia-smi` is not on PATH. Transformers is installed, but `qwen_vl_utils` is not. This does not establish that no GPU exists anywhere on the machine; it does establish that the inspected Python environment currently has no available CUDA device.

Model parameter counts give a rough lower bound for weights alone: about 6 GB at 16-bit for 3B parameters, about 14 GB for 7B, and about 144 GB for 72B. Runtime memory, visual encoder state, image tokens, and generation add overhead. These are arithmetic estimates, not benchmarks. On the inspected CPU-only setup, even the 3B option should not be assumed comfortable or fast; 7B and 72B are not practical assumptions without a suitable GPU or other hardware assessment. Quantization can reduce weight memory but requires a compatible runtime and may affect quality.

Hosted Model Studio inference avoids local weight storage and offers vision endpoints; its documentation describes model-based billing. Uploading page images would disclose their contents to that service, and usage is billable according to the selected endpoint and current pricing. Check the [current pricing documentation](https://help.aliyun.com/zh/model-studio/model-training-and-deployment-billing) before any future paid trial. Region, account, endpoint name, pricing, retention, and availability must be confirmed at the time of use. No API availability or price was tested in this project.

## Recommendation

Retain PyMuPDF and Tesseract as the default. If a later experiment is approved, send only a small set of selected rendered pages to a vision model, after ordinary extraction has been retained. Use the model to return a concise, evidence-oriented description of figure/table content and the locations of labels; ask it to mark unreadable text as uncertain. Store its output in a separate interpretation field with model identifier and prompt/version. Do not replace extracted text, and do not treat the generated description as verified fact until compared with the page.

Do not send every PDF page to Qwen. In the inspected samples, useful candidate pages are book1 PDF pages 31 and 53 (table/diagram and figure/order trouble), book2 PDF pages 32 and 65 (table and mixed prose/diagram), and book2 PDF page 112 (two-column order issue). A sparse section divider such as book1 page 23 is a useful control for deciding whether vision is even needed; it does not appear to need a vision model.

## Proposed small comparison experiment (not run)

If separately approved, compare **five** page images against the existing PyMuPDF/Tesseract result: book1 PDF pages 31 and 53, and book2 PDF pages 32, 65, and 112. Include book1 page 23 as a low-text control only if the experiment has room. Start with one page per model/API request and a fixed prompt asking for (a) visible table cells or diagram labels, (b) spatial/column grouping, (c) uncertain content, and (d) no unsupported conclusions.

For each page, manually score:

1. Coverage of visible headings, labels, and table rows/columns.
2. Correct association of labels/cells and reading order.
3. Exactness of numbers and technical terms.
4. Unsupported statements or invented relationships.
5. Time, input/output tokens or bill, and any image disclosure required.

Compare to the rendered PDF page, not only to PyMuPDF. Keep all model output separate. A single small sample can show whether Qwen is promising for these page types, but cannot establish general accuracy across either book.

## What has and has not been done

- **Verified:** the model family and public run paths described by the linked official sources; the local Python environment’s CPU-only PyTorch status; representative visual extraction failures recorded in the audit report.
- **Not tested:** local model loading, inference speed, GPU detection outside this environment, hosted API availability, privacy/retention terms for a selected account/region, cost for any actual page request, or output quality from Qwen on these PDFs.
- **Not integrated:** no Qwen package, model weights, API call, or change to the extraction pipeline.
