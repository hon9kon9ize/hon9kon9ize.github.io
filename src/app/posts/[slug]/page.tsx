import fs from "fs"
import { ParsedUrlQuery } from "node:querystring"
import matter from "gray-matter"
import hljs from "highlight.js"
import MarkdownIt from "markdown-it"

import "highlight.js/styles/atom-one-dark.css"

interface PathsParams extends ParsedUrlQuery {
  slug: string
}

export async function generateStaticParams() {
  const files = fs.readdirSync("public/posts")
  const posts = files.map((fileName) => {
    const slug = fileName.replace(".md", "")
    const readFile = fs.readFileSync(`public/posts/${fileName}`, "utf-8")
    const { data: frontmatter } = matter(readFile)

    return {
      slug,
      frontmatter,
    }
  })

  return posts.map(({ slug }) => ({
    slug,
  }))
}

export default function Post(context: { params: PathsParams }) {
  const { slug } = context.params as PathsParams
  const fileName = fs.readFileSync(`public/posts/${slug}.md`, "utf-8")
  const { data: metadata, content } = matter(fileName)
  const md: MarkdownIt = new MarkdownIt({
    highlight: function (str, lang) {
      if (lang && hljs.getLanguage(lang)) {
        try {
          return (
            '<pre><code class="hljs" style="background-color: transparent">' +
            hljs.highlight(str, { language: lang, ignoreIllegals: true })
              .value +
            "</code></pre>"
          )
        } catch (__) {}
      }

      return (
        '<pre><code class="hljs" style="background-color: transparent">' +
        md.utils.escapeHtml(str) +
        "</code></pre>"
      )
    },
  })

  return (
    <div className="prose prose-slate dark:prose-invert mx-auto mt-12 flex w-full flex-col gap-1 md:gap-2">
      <h1 className="mb-0">{metadata.title}</h1>
      <p className="m-0 opacity-30">by {metadata.author}</p>
      <p className="m-0 text-xs opacity-30">{metadata.updated}</p>
      <div dangerouslySetInnerHTML={{ __html: md.render(content) }} />
    </div>
  )
}
