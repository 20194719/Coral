"""LLM-backed input and online judgment adapter."""

from src.inference.judges import Judge, LLMProbe


class Backend(Judge):
    def __init__(self, *args, oracle=False, **kwargs):
        """Initialize model state."""
        super().__init__(*args, **kwargs)
        self.oracle = oracle

    def inputs(self, root, view, tables):
        """Prepare dataset judgment inputs."""
        if self.oracle:
            from src.inference.oracle import Oracle

            return Oracle().inputs(root, view, tables)
        return {}, None, None

    def probe(self, view, condition, filters):
        """Build an online judgment function."""
        if self.oracle:
            from src.inference.oracle import Oracle

            return Oracle().probe(view, condition, filters)
        return LLMProbe(
            self, view["tables"], view["spec"]["entities"], condition, filters
        )
