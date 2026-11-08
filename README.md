# AI/ML Toolset Archive Collection

This repository is an artifact shelf for reusable AI/ML platform examples. The current default-branch inventory is:

| Artifact | Format | Notes |
|---|---|---|
| `adaptive-ai-platform.tar.gz` | gzip tar archive | Packaged platform artifact; inspect its archive manifest before use |
| `aiml-observability-stack-fixed.zip` | ZIP | Packaged observability artifact |
| `aws_mlops_eks_github.tar.gz` | gzip tar archive | Packaged AWS/EKS MLOps artifact |
| `cloud_agnostic_ml_inference_platform.tar.gz` | gzip tar archive | Packaged cloud-agnostic inference artifact |
| `helm_ml_ai_workload_catalog.tar.gz` | gzip tar archive | Packaged Helm workload catalog |
| `python_ai_lab_single_pane.tar.gz` | gzip tar archive | Packaged Python AI lab artifact |
| `tensorrt_llm_interactive_code_lab.py` | Python | Interactive TensorRT-LLM learning lab |

The descriptions above are based on filenames and repository inventory. They are not a code review or a claim that the archives have been extracted, tested, or security-scanned.

## Verify archive integrity

GitHub Actions checks that each compressed artifact can be listed/read and that the Python file compiles. It does not execute deployment scripts or provision cloud resources.

To inspect an archive locally:

```bash
tar -tzf adaptive-ai-platform.tar.gz | head
unzip -l aiml-observability-stack-fixed.zip
```

Before using an archive, inspect its source, dependencies, image references, credentials handling, Terraform plans, and teardown scripts. Treat each extracted artifact as untrusted until reviewed.

## Portfolio ownership

Archival shelf for packaged AI/ML platform examples and a TensorRT-LLM learning script. Archive integrity checks do not certify runtime behavior or security.

Account-wide ownership and cleanup decisions are maintained in the [WSP portfolio repository catalog](https://github.com/zeiglerit-aiml/WSP_Integrated_AI_SDLC_GVC_pl/blob/main/REPOSITORY_CATALOG.md).
