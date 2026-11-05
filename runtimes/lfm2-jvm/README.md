# LFM2 JVM cartridge runtime

Shared `llm/v1` adapter, framed persistent JVM worker and POSIX native C ABI source for LFM2 CPU and Vulkan recipes. The profile types come from `:profiles:llm-v1`; the model, tokenizer and generation implementations come from SKaiNET-transformers. This module contains no model weights or built native library.

`check` exercises profile streaming/lifecycle behavior and the actual native bridge using a synthetic worker. `installDist` builds the worker library directory. Blueprint tasks compile `native/build.py` against their exact descriptor bytes and package the resulting shared library as the runtime artifact; JVM JARs are ordinary dependencies.

The former CPU artifact closure remains captured separately. Moving or rebuilding sources does not inherit its target measurement qualification.
