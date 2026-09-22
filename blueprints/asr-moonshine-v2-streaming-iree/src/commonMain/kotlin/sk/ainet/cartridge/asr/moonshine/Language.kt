package sk.ainet.cartridge.asr.moonshine

/** Language selection against what a pack holds: the BCP-47 subtag of [tag], if the pack has it, else [fallback]. */
public object Language {
    public fun subtag(tag: String): String = tag.substringBefore('-').substringBefore('_').lowercase()

    /** The flavor to load for [tag] from [available]; `null` when the pack has neither the tag's language nor [fallback]. */
    public fun select(tag: String, available: List<String>, fallback: String = "en"): String? {
        val wanted = subtag(tag)
        return when {
            wanted in available -> wanted
            fallback in available -> fallback
            else -> null
        }
    }
}
