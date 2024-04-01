export const SERVICES_SUMMARY =
  "We provide model fine-tuning service based on our language model. We also provide consulting services for your NLP projects."

export const Services: React.FC = () => (
  <div className="flex flex-col gap-4 lg:gap-8">
    <h1 className="text-3xl">Services</h1>
    <p className="text-base">
      We offer comprehensive services tailored to enhance your NLP projects. Our
      expertise lies in providing model fine-tuning solutions utilizing our
      advanced language model. Additionally, we offer consulting services to
      guide you through the intricacies of your NLP endeavors.
    </p>

    <div className="flex flex-col gap-4">
      <h2 className="text-2xl">Model Fine-tuning</h2>
      <p className="text-base">
        we understand that no single language model can fully meet the unique
        requirements of your business. That&apos;s why we offer a specialized
        model fine-tuning service, leveraging our advanced language model, to
        precisely tailor the solution to your specific needs.
      </p>
    </div>

    <div className="flex flex-col gap-4">
      <h2 className="text-2xl">Model Deployment and Integration</h2>
      <p className="text-base">
        We provide model deployment and integration services to ensure that your
        NLP projects are seamlessly integrated into your existing
        infrastructure. Our experts will guide you through the process to ensure
        a smooth transition.
      </p>
    </div>
  </div>
)
