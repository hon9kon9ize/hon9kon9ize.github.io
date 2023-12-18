import fs from "fs"
import { Metadata } from "next"
import Image from "next/image"
import Link from "next/link"
import matter from "gray-matter"

export const metadata: Metadata = {
  title: "Blog | hon9kon9ize",
}

const ItemArrow: React.FC = () => (
  <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
    -&gt;
  </span>
)

export default function Blog() {
  const files = fs.readdirSync("public/posts")
  const posts = files
    .map((fileName) => {
      const slug = fileName.replace(".md", "")
      const readFile = fs.readFileSync(`public/posts/${fileName}`, "utf-8")
      const { data: frontmatter } = matter(readFile)

      return {
        slug,
        frontmatter,
      }
    })
    .sort((a, b) => {
      if (a.frontmatter.updated > b.frontmatter.updated) {
        return -1
      } else if (a.frontmatter.updated < b.frontmatter.updated) {
        return 1
      } else {
        return 0
      }
    })

  return (
    <div className="flex-col pt-12">
      <h1 className="text-center text-3xl md:text-left">Blogs</h1>
      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 md:p-0 lg:grid-cols-3">
        {posts.map(({ slug, frontmatter }) => (
          <div key={slug} className="m-2 flex flex-col">
            <Link href={`/posts/${slug}`} legacyBehavior>
              <a className="group rounded-lg border  border-gray-400 px-4 py-3 transition-colors hover:bg-gray-100 dark:border-neutral-700 hover:dark:bg-neutral-800/30">
                <div className="flex h-full flex-col gap-2">
                  <Image
                    src={frontmatter.image}
                    width="320"
                    height="168"
                    alt={frontmatter.title}
                    className="min-w-full"
                  />
                  <h4 className="p-0">
                    {frontmatter.title} <ItemArrow />
                  </h4>
                  <p className="m-0 text-xs opacity-30">
                    {frontmatter.updated}
                  </p>
                  <p className="m-0 flex-1 truncate text-sm opacity-50">
                    {frontmatter.description}
                  </p>
                </div>
              </a>
            </Link>
          </div>
        ))}
      </div>
    </div>
  )
}
