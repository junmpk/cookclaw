/**
 * 流式 Markdown 过滤器 — 字符级状态机，去除微信不支持的 Markdown 语法
 *
 * 保留的语法：代码围栏(```)、行内代码(`)、表格、水平线、粗体(**)
 * 过滤的语法：斜体包裹中文、H5/H6 标题、图片标记
 */
export class StreamingMarkdownFilter {
  private buf = "";
  private fence = false;
  private sol = true;
  private inl: { type: "image" | "bold3" | "italic" | "ubold3" | "uitalic"; acc: string } | null = null;

  feed(delta: string): string {
    this.buf += delta;
    return this.pump(false);
  }

  flush(): string {
    return this.pump(true);
  }

  private pump(eof: boolean): string {
    let out = "";
    while (this.buf) {
      const sLen = this.buf.length;
      const sSol = this.sol;
      const sFence = this.fence;
      const sInl = this.inl;

      if (this.fence) out += this.pumpFence(eof);
      else if (this.inl) out += this.pumpInline(eof);
      else if (this.sol) out += this.pumpSOL(eof);
      else out += this.pumpBody(eof);

      if (this.buf.length === sLen && this.sol === sSol &&
          this.fence === sFence && this.inl === sInl) break;
    }

    if (eof && this.inl) {
      const markers: Record<string, string> = { image: "![", bold3: "***", italic: "*", ubold3: "___", uitalic: "_" };
      out += (markers[this.inl.type] ?? "") + this.inl.acc;
      this.inl = null;
    }
    return out;
  }

  private pumpFence(eof: boolean): string {
    if (this.sol) {
      if (this.buf.length < 3 && !eof) return "";
      if (this.buf.startsWith("```")) {
        const nl = this.buf.indexOf("\n", 3);
        if (nl !== -1) {
          this.fence = false;
          const line = this.buf.slice(0, nl + 1);
          this.buf = this.buf.slice(nl + 1);
          this.sol = true;
          return line;
        }
        if (eof) {
          this.fence = false;
          const line = this.buf;
          this.buf = "";
          return line;
        }
        return "";
      }
      this.sol = false;
    }
    const nl = this.buf.indexOf("\n");
    if (nl !== -1) {
      const chunk = this.buf.slice(0, nl + 1);
      this.buf = this.buf.slice(nl + 1);
      this.sol = true;
      return chunk;
    }
    const chunk = this.buf;
    this.buf = "";
    return chunk;
  }

  private pumpSOL(eof: boolean): string {
    const b = this.buf;

    if (b[0] === "\n") { this.buf = b.slice(1); return "\n"; }

    if (b[0] === "`") {
      if (b.length < 3 && !eof) return "";
      if (b.startsWith("```")) {
        const nl = b.indexOf("\n", 3);
        if (nl !== -1) {
          this.fence = true;
          const line = b.slice(0, nl + 1);
          this.buf = b.slice(nl + 1);
          this.sol = true;
          return line;
        }
        if (eof) { this.buf = ""; return b; }
        return "";
      }
      this.sol = false;
      return "";
    }

    if (b[0] === ">") { this.sol = false; return ""; }

    if (b[0] === "#") {
      let n = 0;
      while (n < b.length && b[n] === "#") n++;
      if (n === b.length && !eof) return "";
      if (n >= 5 && n <= 6 && n < b.length && b[n] === " ") {
        this.buf = b.slice(n + 1);
        this.sol = false;
        return "";
      }
      this.sol = false;
      return "";
    }

    if (b[0] === " " || b[0] === "\t") {
      if (b.search(/[^ \t]/) === -1 && !eof) return "";
      this.sol = false;
      return "";
    }

    if (b[0] === "-" || b[0] === "*" || b[0] === "_") {
      const ch = b[0];
      let j = 0;
      while (j < b.length && (b[j] === ch || b[j] === " ")) j++;
      if (j === b.length && !eof) return "";
      if (j === b.length || b[j] === "\n") {
        let count = 0;
        for (let k = 0; k < j; k++) if (b[k] === ch) count++;
        if (count >= 3) {
          if (j < b.length) { this.buf = b.slice(j + 1); this.sol = true; return b.slice(0, j + 1); }
          this.buf = "";
          return b;
        }
      }
      this.sol = false;
      return "";
    }

    this.sol = false;
    return "";
  }

  private pumpBody(eof: boolean): string {
    let out = "";
    let i = 0;
    while (i < this.buf.length) {
      const c = this.buf[i];
      if (c === "\n") {
        out += this.buf.slice(0, i + 1);
        this.buf = this.buf.slice(i + 1);
        this.sol = true;
        return out;
      }
      if (c === "!" && i + 1 < this.buf.length && this.buf[i + 1] === "[") {
        out += this.buf.slice(0, i);
        this.buf = this.buf.slice(i + 2);
        this.inl = { type: "image", acc: "" };
        return out;
      }
      if (c === "~" && i + 1 < this.buf.length && this.buf[i + 1] === "~") {
        out += this.buf.slice(0, i);
        this.buf = this.buf.slice(i + 2);
        return out;
      }
      if (c === "*" && i + 2 < this.buf.length && this.buf[i + 1] === "*" && this.buf[i + 2] === "*") {
        out += this.buf.slice(0, i);
        this.buf = this.buf.slice(i + 3);
        this.inl = { type: "bold3", acc: "" };
        return out;
      }
      if (c === "_" && i + 2 < this.buf.length && this.buf[i + 1] === "_" && this.buf[i + 2] === "_") {
        out += this.buf.slice(0, i);
        this.buf = this.buf.slice(i + 3);
        this.inl = { type: "ubold3", acc: "" };
        return out;
      }
      if (c === "*") {
        out += this.buf.slice(0, i);
        this.buf = this.buf.slice(i + 1);
        this.inl = { type: "italic", acc: "" };
        return out;
      }
      if (c === "_") {
        out += this.buf.slice(0, i);
        this.buf = this.buf.slice(i + 1);
        this.inl = { type: "uitalic", acc: "" };
        return out;
      }
      i += 1;
    }
    if (eof) {
      out += this.buf;
      this.buf = "";
    }
    return out;
  }

  private pumpInline(eof: boolean): string {
    if (!this.inl) return "";
    const b = this.buf;

    if (this.inl.type === "image") {
      const close = b.indexOf("](");
      if (close === -1) {
        if (eof) {
          const acc = this.inl.acc + b;
          this.buf = "";
          this.inl = null;
          return `![${acc}`;
        }
        this.inl.acc += b;
        this.buf = "";
        return "";
      }
      const paren = b.indexOf(")", close + 2);
      if (paren === -1) {
        if (eof) {
          const acc = this.inl.acc + b;
          this.buf = "";
          this.inl = null;
          return `![${acc}`;
        }
        this.inl.acc += b;
        this.buf = "";
        return "";
      }
      this.buf = b.slice(paren + 1);
      this.inl = null;
      return "";
    }

    const markerLen = this.inl.type === "bold3" || this.inl.type === "ubold3" ? 3 : 1;
    const marker = this.inl.type === "bold3" ? "***" : this.inl.type === "ubold3" ? "___" : this.inl.type === "italic" ? "*" : "_";

    const idx = b.indexOf(marker);
    if (idx !== -1) {
      const content = b.slice(0, idx);
      this.buf = b.slice(idx + markerLen);
      const acc = this.inl.acc + content;

      if (this.inl.type === "italic" || this.inl.type === "uitalic") {
        const hasCJK = /[\u4e00-\u9fff\u3400-\u4dbf]/.test(acc);
        if (hasCJK) {
          this.inl = null;
          return acc;
        }
        this.inl = null;
        return marker + acc + marker;
      }

      this.inl = null;
      return acc;
    }

    if (eof) {
      const acc = this.inl.acc + b;
      this.buf = "";
      this.inl = null;
      return marker + acc;
    }

    this.inl.acc += b;
    this.buf = "";
    return "";
  }
}
