import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const blogPosts = [
  {
    title: "The Rise of AI in Healthcare",
    desc: "Discover how artificial intelligence is revolutionizing medical diagnostics and patient care.",
    img: "https://imgs.search.brave.com/MztujIlfj_2vwXAnP1wNTq8xj5qkA7p_0CXTK_U70DA/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5pc3RvY2twaG90/by5jb20vaWQvMTE3/NjA2NzI2Ni9waG90/by9hdXRoZW50aWNh/dGlvbi1ieS1mYWNp/YWwtcmVjb2duaXRp/b24tY29uY2VwdC1i/aW9tZXRyaWMtc2Vj/dXJpdHktc3lzdGVt/LmpwZz9zPTYxMng2/MTImdz0wJms9MjAm/Yz1sdlN2UDJfVUJl/ekhucXhjcngxY3Vf/bExCYUJoYTVMQlZn/Nkh3VVBxT0xRPQ",
    content: `
      <p>Artificial Intelligence (AI) is rapidly transforming the landscape of healthcare, moving beyond theoretical concepts into practical applications that save lives and improve patient outcomes. From advanced diagnostics to personalized treatment plans, AI's capabilities are enhancing every aspect of medical practice.</p>
      
      <h3>Revolutionizing Diagnostics</h3>
      <p>AI algorithms, particularly deep learning, are proving incredibly adept at analyzing medical images such as X-rays, MRIs, and CT scans with remarkable accuracy, often matching or even surpassing human radiologists. This can lead to earlier and more precise detection of diseases like cancer, enabling timely interventions.</p>
      
      <h3>Personalized Treatment Plans</h3>
      <p>One of the most exciting applications of AI in healthcare is its ability to tailor treatment plans to individual patients. By analyzing a patient's genetic makeup, medical history, lifestyle, and response to previous treatments, AI can predict the most effective therapeutic strategies, minimizing trial-and-error and improving efficacy.</p>
      
      <h3>Drug Discovery and Development</h3>
      <p>The process of discovering new drugs is notoriously long and expensive. AI is accelerating this by identifying potential drug candidates, predicting their efficacy and toxicity, and optimizing molecular structures. This significantly reduces the time and cost associated with bringing new medications to market.</p>
      
      <h3>Challenges and Ethical Considerations</h3>
      <p>Despite its immense potential, the integration of AI into healthcare comes with challenges, including data privacy, regulatory hurdles, and the need for robust validation of AI models. Ethical considerations, such as algorithmic bias and accountability, are also paramount to ensure equitable and responsible deployment.</p>
      
      <p>As AI continues to evolve, its collaborative potential with human medical professionals will be key to unlocking a future of more efficient, accurate, and personalized healthcare for all.</p>
    `,
  },
  {
    title: "Top 5 Health Trends in 2025",
    desc: "From wearable tech to AI-driven diets, here’s what’s trending in health.",
    img: "https://imgs.search.brave.com/PXSfHb4OCH2U2a26m_pn6EeMW4KqIiXm_rIscOW3TF8/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5pc3RvY2twaG90/by5jb20vaWQvMTQ2/NjkwMTE5My9waG90/by9iaW9tZXRyaWMt/c2VjdXJpdHktc2Nh/bi5qcGc_cz02MTJ4/NjEyJnc9MCZrPTIw/JmM9eldJREI0NjBC/TFpiTmlXMGVfS2p2/ZUhCN1FOcmNKN19Y/Sk9uc0JDMXZfMD0",
    content: `
      <p>The year 2025 is shaping up to be a pivotal year for health innovation, with several trends poised to redefine how we manage our well-being. Here are the top 5 health trends to watch:</p>

      <h3>1. Hyper-Personalized Nutrition</h3>
      <p>Forget one-size-fits-all diets. Advances in genomics and microbiome research, combined with AI, are leading to nutrition plans tailored precisely to an individual's unique biological makeup and health goals. Expect more at-home testing kits that inform AI-powered meal recommendations.</p>

      <h3>2. Advanced Wearable Technology</h3>
      <p>Wearables are evolving beyond step counting. New devices will offer continuous, non-invasive monitoring of a wider array of biomarkers, including glucose levels, hydration, and even early signs of infection. These insights will empower proactive health management.</p>

      <h3>3. Mental Wellness Integration</h3>
      <p>Mental health is finally receiving the attention it deserves, with a greater integration of mental wellness strategies into mainstream healthcare. Expect more AI-driven therapy apps, virtual reality for stress reduction, and accessible digital platforms for mental health support.</p>

      <h3>4. Telemedicine 2.0</h3>
      <p>Building on the acceleration seen during the pandemic, telemedicine will become even more sophisticated. Enhanced virtual diagnostic tools, remote monitoring devices, and AI-powered symptom checkers will make virtual consultations more comprehensive and effective, bridging geographical gaps in care.</p>

      <h3>5. Proactive Preventative Healthcare</h3>
      <p>The focus is shifting from treating illness to preventing it. AI-powered predictive analytics, fueled by vast health datasets, will help individuals and healthcare providers identify potential health risks long before symptoms appear. This trend emphasizes lifestyle interventions, early screenings, and personalized risk assessments.</p>

      <p>These trends highlight a future where health is increasingly personalized, preventive, and powered by technology, empowering individuals to take greater control of their well-being.</p>
    `,
  },
  {
    title: "Preventive Care: Why It Matters",
    desc: "Learn why prevention is better than cure in modern medicine and how it saves lives.",
    img: "https://imgs.search.brave.com/tA-xXHz412UIQAgs3n82SsMfJTdnMPMAF16G-TxmgM0/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5pc3RvY2twaG90/by5jb20vaWQvMTM5/NjgwMTg2OC9waG90/by9hLXlvdW5nLWFm/cmljYW4tYW1lcmlj/YW4tZG9jdG9yLXdv/cmtzLW9uLWh1ZC1v/ci1ncmFwaGljLWRp/c3BsYXktaW4tZnJv/bnQtb2YtaGVyLmpw/Zz9zPTYxMng2MTIm/dz0wJms9MjAmYz1s/b2pZaU5KUzZPcEgy/NG1PYTFYaGF0My1S/Rm52b2lsUkJkZXBD/UE5xMXNFPQ",
    content: `
      <p>In the complex world of modern medicine, a fundamental truth often gets overlooked: prevention is not just better than cure, it's often more effective and sustainable in the long run. Preventive care focuses on maintaining health and avoiding disease rather than just treating conditions once they arise.</p>

      <h3>Cost-Effectiveness</h3>
      <p>One of the most compelling arguments for preventive care is its economic benefit. Detecting and addressing health issues early, or preventing them entirely through vaccinations, healthy lifestyle choices, and regular screenings, can significantly reduce the need for costly hospitalizations, complex treatments, and long-term medication.</p>

      <h3>Improved Quality of Life</h3>
      <p>Beyond financial savings, preventive care directly impacts an individual's quality of life. By preventing chronic diseases like diabetes, heart disease, and certain cancers, people can enjoy longer, healthier, and more active lives, free from the burdens of managing serious health conditions.</p>

      <h3>Key Pillars of Preventive Care</h3>
      <ul>
        <li><strong>Regular Check-ups:</strong> Annual physicals and screenings can catch early signs of disease.</li>
        <li><strong>Vaccinations:</strong> Immunizations protect against infectious diseases.</li>
        <li><strong>Healthy Lifestyle:</strong> Balanced diet, regular exercise, adequate sleep, and stress management are crucial.</li>
        <li><strong>Risk Assessment:</strong> Understanding personal and family medical history to identify predispositions.</li>
        <li><strong>Education:</strong> Being informed about health risks and healthy practices.</li>
      </ul>

      <h3>The Role of Technology</h3>
      <p>Technology, especially AI and wearable devices, is playing an increasingly vital role in preventive care. These tools can monitor health metrics, identify patterns, and even predict potential health issues, empowering individuals to make informed decisions and take proactive steps.</p>

      <p>Embracing a preventive mindset is an investment in your future well-being, leading to a healthier population and a more sustainable healthcare system.</p>
    `,
  },
  {
    title: "Understanding Your Blood Test Results",
    desc: "Demystifying common blood test markers and what they mean for your health.",
    img: "https://imgs.search.brave.com/HiGmUJ96Y8zdjT_Avf3kBiKXtXEHGnWm2j32ErGWa5M/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5pc3RvY2twaG90/by5jb20vaWQvOTQz/MDY3NDYwL3Bob3Rv/L21hbGUtaXQtc3Bl/Y2lhbGlzdC1ob2xk/cy1sYXB0b3AtYW5k/LWRpc2N1c3Nlcy13/b3JrLXdpdGgtZmVt/YWxlLXNlcnZlci10/ZWNobmljaWFuLXRo/ZXlyZS5qcGc_cz02/MTJ4NjEyJnc9MCZr/PTIwJmM9ODUxQXJt/RjJvb3otMnlRQ1JD/V2tqSkxDWXdEZHBU/Q1l6UGlubDlXZ0Ff/cz0",
    content: `
      <p>Blood tests are a fundamental part of routine healthcare, offering a powerful snapshot of your internal health. However, understanding the myriad of markers and what they signify can be daunting. Let's demystify some common blood test components:</p>

      <h3>Complete Blood Count (CBC)</h3>
      <ul>
        <li><strong>Red Blood Cells (RBC):</strong> Carry oxygen. High or low levels can indicate conditions like anemia or dehydration.</li>
        <li><strong>White Blood Cells (WBC):</strong> Part of the immune system. Abnormal levels can signal infection, inflammation, or immune disorders.</li>
        <li><strong>Platelets:</strong> Involved in blood clotting. Issues can lead to excessive bleeding or clotting.</li>
      </ul>

      <h3>Lipid Panel</h3>
      <ul>
        <li><strong>Total Cholesterol:</strong> Overall cholesterol level.</li>
        <li><strong>LDL ("Bad" Cholesterol):</strong> High levels increase heart disease risk.</li>
        <li><strong>HDL ("Good" Cholesterol):</strong> Higher levels protect against heart disease.</li>
        <li><strong>Triglycerides:</strong> High levels can increase heart disease risk.</li>
      </ul>

      <h3>Blood Glucose Test</h3>
      <p>Measures the sugar (glucose) level in your blood. Crucial for detecting and monitoring diabetes or prediabetes. Fasting glucose, HbA1c, and oral glucose tolerance tests are common variations.</p>

      <h3>Liver Function Tests (LFTs)</h3>
      <p>A group of tests that measure enzymes and proteins in your blood that are produced by the liver. Abnormal results can indicate liver damage or disease.</p>

      <h3>Kidney Function Tests</h3>
      <p>Measures substances like creatinine and blood urea nitrogen (BUN) to assess how well your kidneys are filtering waste from your blood.</p>

      <p>Always discuss your blood test results with your doctor. They can interpret the findings in the context of your overall health, symptoms, and medical history, providing personalized advice and further steps if needed.</p>
    `,
  },
  {
    title: "The Future of Personalized Medicine",
    desc: "How genomics and AI are shaping highly individualized treatments.",
    img: "https://imgs.search.brave.com/pJ_kPV4j8uAg0x_EeGEgBH9Qbg5WeKwXO0YIeilWZfs/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly93d3cu/YWhhLm9yZy9zaXRl/cy9kZWZhdWx0L2Zp/bGVzLzIwMjQtMTIv/QUlfY292ZXJfNzAw/eDUzMi5qcGc",
    content: `
      <p>Personalized medicine, also known as precision medicine, is an evolving field that aims to tailor medical decisions, treatments, practices, and products to the individual patient based on their predicted response or risk of disease. This approach stands in contrast to the "one-size-fits-all" model of medicine.</p>

      <h3>Genomics at the Core</h3>
      <p>The rise of affordable and comprehensive genomic sequencing is a cornerstone of personalized medicine. By understanding an individual's unique genetic code, doctors can identify predispositions to certain diseases, predict how they might react to specific drugs, and even design therapies that target specific genetic mutations.</p>

      <h3>AI as the Enabler</h3>
      <p>While genomics provides the raw data, Artificial Intelligence acts as the powerful engine that makes personalized medicine scalable and actionable. AI algorithms can analyze vast datasets of genetic information, clinical records, and research findings to:</p>
      <ul>
        <li>Identify complex patterns and correlations.</li>
        <li>Predict disease progression and treatment outcomes.</li>
        <li>Suggest optimal drug dosages or combinations.</li>
        <li>Discover new biomarkers for early detection.</li>
      </ul>

      <h3>Targeted Therapies</h3>
      <p>In oncology, personalized medicine is already transforming cancer treatment. Genetic testing of tumors allows for the selection of targeted therapies that specifically attack cancer cells with particular genetic abnormalities, sparing healthy cells and often leading to better results with fewer side effects.</p>

      <h3>Preventive Strategies</h3>
      <p>Beyond treatment, personalized medicine also empowers proactive preventive strategies. Knowing one's genetic risks can motivate lifestyle changes, earlier screenings, or specific preventive interventions to mitigate potential health challenges before they manifest.</p>

      <p>The future of medicine is highly individualized. As genomics and AI continue to advance, personalized medicine promises a new era of healthcare that is more precise, effective, and tailored to the unique needs of every patient.</p>
    `,
  },
  {
    title: "Mindfulness for Better Health",
    desc: "Simple mindfulness techniques to reduce stress and improve well-being.",
    img: "https://imgs.search.brave.com/8WswVb8nh-6aBxOxwodg2GMFfSipJaLrXzLgmrxYLMQ/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5pc3RvY2twaG90/by5jb20vaWQvMTQ3/MDIxOTUxNy9waG90/by9kb2N0b3Itc3Vy/Z2Vvbi1hbmQtbmV1/cm9sb2dpc3QtdXNl/LXJvYm90aWMtYW5k/LW1lZGljYWwtdGVj/aG5vbG9neS1kaWFn/bm9zZS1hbmQtZXhh/bWluZS5qcGc_cz02/MTJ4NjEyJnc9MCZr/PTIwJmM9elBtSTlX/dlNjRmJpTV9qekhx/M1czdVNRRDAyZG5X/ZjJHRHM4NHUxZGw3/cz0",
    content: `
      <p>In our fast-paced world, stress has become a pervasive companion. Mindfulness, the practice of being present and fully engaged in the current moment without judgment, offers a powerful antidote. Beyond mental tranquility, mindfulness has profound benefits for physical health too.</p>

      <h3>What is Mindfulness?</h3>
      <p>Mindfulness is essentially paying attention, on purpose, to the present moment, and doing so non-judgmentally. It involves observing your thoughts, feelings, and bodily sensations as they arise, rather than getting caught up in them or reacting automatically.</p>

      <h3>Health Benefits of Mindfulness</h3>
      <ul>
        <li><strong>Stress Reduction:</strong> Mindfulness practices can lower cortisol levels (the stress hormone) and reduce symptoms of anxiety and depression.</li>
        <li><strong>Improved Sleep:</strong> By calming the mind, mindfulness can help combat insomnia and lead to more restful sleep.</li>
        <li><strong>Enhanced Emotional Regulation:</strong> It helps you respond to challenging situations more thoughtfully rather than reactively.</li>
        <li><strong>Boosted Immune Function:</strong> Some studies suggest a link between mindfulness and a stronger immune system.</li>
        <li><strong>Better Focus and Concentration:</strong> Regular practice can improve attention span and cognitive function.</li>
      </ul>

      <h3>Simple Mindfulness Techniques to Try</h3>
      <ol>
        <li><strong>Mindful Breathing:</strong> Sit comfortably, close your eyes, and focus solely on the sensation of your breath as it enters and leaves your body. When your mind wanders, gently bring it back to your breath.</li>
        <li><strong>Body Scan Meditation:</strong> Lie down and bring your attention sequentially to different parts of your body, noticing any sensations without judgment.</li>
        <li><strong>Mindful Eating:</strong> Pay full attention to the taste, texture, smell, and appearance of your food, eating slowly and savoring each bite.</li>
        <li><strong>Walking Meditation:</strong> As you walk, bring your awareness to the sensations of your feet touching the ground, the movement of your legs, and your surroundings.</li>
      </ol>

      <p>Incorporating even a few minutes of mindfulness into your daily routine can lead to significant improvements in both your mental and physical health. It's a simple yet profound practice that empowers you to live more fully in the present.</p>
    `,
  },
];


const Blogs = () => {
  const [selectedBlog, setSelectedBlog] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false); // Corrected state variable name

  // Close modal on Escape key press
  useEffect(() => {
    const handleEscape = (event) => {
      if (event.key === "Escape" && isModalOpen) {
        closeModal();
      }
    };
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("keydown", handleEscape);
    };
  }, [isModalOpen]);

  const openModal = (blog) => {
    setSelectedBlog(blog);
    setIsModalOpen(true);
    document.body.style.overflow = 'hidden'; // Prevent scrolling when modal is open
  };

  const closeModal = () => {
    setIsModalOpen(false); // Corrected function call
    setSelectedBlog(null);
    document.body.style.overflow = 'unset'; // Re-enable scrolling
  };

  // Animation variants for modal
  const modalVariants = {
    hidden: { opacity: 0, scale: 0.95 },
    visible: { opacity: 1, scale: 1, transition: { duration: 0.3, ease: "easeOut" } },
    exit: { opacity: 0, scale: 0.95, transition: { duration: 0.2, ease: "easeIn" } },
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background Video */}
      <video
        className="fixed top-0 left-0 w-full h-full object-cover z-[-1]"
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
        poster="https://i.pinimg.com/videos/thumbnails/originals/2e/fc/a0/2efca0faa924c69e71e576da04168958.0000000.jpg"
        src="https://cdn.pixabay.com/video/2023/04/02/157741-818722985_large.mp4"
      ></video>

      {/* Overlay */}
      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-[#0d1b2a]/80 to-[#1b263b]/90 z-0"></div>

      <div className="relative z-10 px-4 py-12 sm:px-6 lg:px-8 text-white">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
            className="text-center mb-16"
          >
            <h1 className="text-4xl sm:text-5xl font-bold text-blue-300 mb-4">HealthPredict Blog</h1>
            <p className="max-w-2xl mx-auto text-blue-100 text-lg">
              Dive into the latest in health tech, AI in medicine, and wellness tips from our team.
            </p>
          </motion.div>

          {/* Blog Cards */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.2 }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8"
          >
            {blogPosts.map((blog, idx) => (
              <motion.div
                key={idx}
                whileHover={{ scale: 1.03 }}
                className="bg-[#1e3a5f]/70 rounded-xl overflow-hidden shadow-lg cursor-pointer transform transition-transform duration-200"
                onClick={() => openModal(blog)} // Open modal on click
              >
                <img src={blog.img} alt={blog.title} className="w-full h-48 object-cover" />
                <div className="p-6">
                  <h3 className="text-xl font-semibold text-blue-200 mb-2">{blog.title}</h3>
                  <p className="text-blue-100">{blog.desc}</p>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>

      {/* Blog Post Modal */}
      <AnimatePresence>
        {isModalOpen && selectedBlog && (
          <motion.div
            className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50 overflow-y-auto" // Added overflow-y-auto here for responsiveness
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={closeModal} // Close modal when clicking outside
          >
            <motion.div
              className="bg-[#1e3a5f] rounded-xl shadow-2xl p-6 w-full max-w-3xl max-h-[90vh] overflow-y-auto relative text-white md:p-8 lg:p-10" // Adjusted padding for larger screens
              variants={modalVariants}
              initial="hidden"
              animate="visible"
            //   style={{ cursor: "pointer" }} // Prevent pointer events on modal content
              exit="exit"
              onClick={(e) => e.stopPropagation()} // Prevent modal from closing when clicking inside
            >
              <button
                onClick={closeModal}
                className="absolute top-4 right-4 text-blue-300 hover:text-white text-3xl font-bold transition-colors duration-200"
                aria-label="Close" // Added for accessibility
              >
                &times;
              </button>
              <img
                src={selectedBlog.img}
                alt={selectedBlog.title}
                className="w-full h-64 object-cover rounded-lg mb-6"
              />
              <h2 className="text-3xl font-bold text-blue-200 mb-4">{selectedBlog.title}</h2>
              <div
                className="prose prose-invert prose-lg max-w-none text-blue-100 leading-relaxed space-y-4" // Added space-y-4 for vertical rhythm
                dangerouslySetInnerHTML={{ __html: selectedBlog.content }}
              ></div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Blogs;