package sk.ainet.cartridge.classification.moonshine.head.recipe

/**
 * Step `export`: the fitted head as one StableHLO function ([HeadExport]).
 *
 *   --weights <head.safetensors> --out <head.mlir>
 */
public fun main(args: Array<String>) {
    val a = Args(args)
    val weights = a.file("weights")
    val config = HeadWeights.config(weights)
    val mlir = HeadExport.stableHlo(config, HeadWeights.read(weights))
    a.file("out").apply { absoluteFile.parentFile.mkdirs() }.writeText(mlir)
    println("export: @${HeadExport.FUNCTION}(rows[1, ${config.maxRows}, ${config.rowWidth}]) -> logits[1, ${config.classes}], ${mlir.length / 1024} KiB")
}
