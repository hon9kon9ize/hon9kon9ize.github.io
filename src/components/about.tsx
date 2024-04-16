export const About: React.FC = () => (
  <div className="flex flex-col gap-4">
    <h1 className="text-3xl">About Us</h1>
    <p className="text-base">
      We are a open community of individuals who are passionate about machine
      learning and Hong Kong culture. In the era of large language models, open
      source LLMs are relying on the community to contribute to dataset and
      model development, if the language or the knowledge of the culture is not
      well represented in the model, the model will not be able to provide
      accurate predictions or generate meaningful text.
    </p>
    <p className="text-base">
      Cantonese, being a low-resource language, has inspired us to contribute
      our extensive knowledge and expertise to the community, with the aim of
      making Cantonese more accessible to all.
    </p>
    <p className="text-base">
      In addition, we are actively seeking like-minded individuals who share our
      enthusiasm for both Cantonese and machine learning. If you are interested
      in joining us on this exciting journey, please do not hesitate to join our
      community.
    </p>
    <a
      className="text-base text-blue-500 hover:underline"
      href="mailto:info@hon9kon9ize.com"
    >
      info@hon9kon9ize.com
    </a>
  </div>
)
