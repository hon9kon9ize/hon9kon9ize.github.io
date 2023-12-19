import fs from "fs"
import { ParsedUrlQuery } from "node:querystring"
import { Metadata, ResolvingMetadata } from "next"
import matter from "gray-matter"
import hljs from "highlight.js"
import MarkdownIt from "markdown-it"
import mia from "markdown-it-anchor"
import mila from "markdown-it-link-attributes"

import "highlight.js/styles/atom-one-dark.css"

/** Same with GitHub */
const slugify = (hash: string) =>
  encodeURIComponent(
    hash
      .trim()
      .toLowerCase()
      // whitespace
      .replace(/ /g, "-")
      // Special symbol
      .replace(
        /[　`~!@#$%^&*()=+\[{\]}\\|;:'",<.>/?·～！¥…（）—【「】」、；：‘“’”，《。》？]/g,
        ""
      )
      // 全角符号
      .replace(/[\uff00-\uffff]/g, "")
  )

interface PathsParams extends ParsedUrlQuery {
  slug: string
}

interface Props {
  params: PathsParams
  searchParams: URLSearchParams
}

export async function generateMetadata(
  { params }: Props,
  parent: ResolvingMetadata
): Promise<Metadata> {
  // read route params
  const slug = params.slug

  // read markdown file
  const fileName = fs.readFileSync(`public/posts/${slug}.md`, "utf-8")
  const { data: metadata } = matter(fileName)
  const previousImages = (await parent).openGraph?.images || []
  const pageMetaImages =
    metadata.image && /^\//.test(metadata.image)
      ? [`https://www.hon9kon9ize.com${metadata.image}`]
      : []

  // return metadata
  return {
    ...parent,
    ...metadata,
    title: `${metadata.title} | Blog | hon9kon9ize`,
    description: metadata.description,
    twitter: {
      cardType: "summary_large_image",
      description: metadata.description,
      image: pageMetaImages[0],
      title: `${metadata.title} | Blog | hon9kon9ize`,
      url: `https://www.hon9kon9ize.com/posts/${slug}`,
      domain: "hon9kon9ize.com",
    },
    openGraph: {
      title: `${metadata.title} | Blog | hon9kon9ize`,
      description: metadata.description,
      images: [...pageMetaImages, ...previousImages],
      url: `https://www.hon9kon9ize.com/posts/${slug}`,
      type: "article",
    },
  } as Metadata
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
    .use(mia, {
      level: [1, 2],
      slugify,
      permalink: true,
      permalinkSpace: false,
      permalinkClass: "anchor",
      permalinkBefore: true,
    })
    .use(mila, {
      matcher(href: string) {
        return href.startsWith("https:")
      },
      attrs: { target: "_blank", rel: "noopener" },
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
