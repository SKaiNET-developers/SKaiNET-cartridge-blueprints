package sk.ainet.cartridge.nlu.functiongemma

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readString
import kotlinx.io.writeString

/** [PrefixIdCache] as one small text file per key under [dir]. Failures are swallowed: a cache must never break a warm-up. */
public class FilePrefixIdCache(private val dir: Path) : PrefixIdCache {
    override fun get(key: String): IntArray? = runCatching {
        val file = Path(dir, "nlu-functiongemma-prefix-$key.ids")
        if (!SystemFileSystem.exists(file)) return null
        SystemFileSystem.source(file).buffered().use { it.readString() }.split(',').map { it.trim().toInt() }.toIntArray()
    }.getOrNull()

    override fun put(key: String, ids: IntArray) {
        runCatching {
            SystemFileSystem.createDirectories(dir)
            SystemFileSystem.sink(Path(dir, "nlu-functiongemma-prefix-$key.ids")).buffered().use { it.writeString(ids.joinToString(",")) }
        }
    }
}
