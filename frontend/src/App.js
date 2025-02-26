import { useState, useEffect, useCallback} from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Twitter } from "lucide-react";



export default function App() {
  const [statement, setStatement] = useState("");
  const [tweetUrl, setTweetUrl] = useState("");
  const [target, setTarget] = useState("");
  const [targets, setTargets] = useState([]);
  const [filteredTargets, setFilteredTargets] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [probabilities, setProbabilities] = useState(null);
  const [history, setHistory] = useState([]);
  const [attentionScores, setAttentionScores] = useState([]);
  const [tokens, setTokens] = useState([]);
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const [error, setError] = useState("");
  const [analysisMode, setAnalysisMode] = useState("text");
  const [isRateLimited, setIsRateLimited] = useState(false);
  const [cooldownTimer, setCooldownTimer] = useState(0);
  const [tweetQueue, setTweetQueue] = useState([]);
  const [isProcessingQueue, setIsProcessingQueue] = useState(false);

  // Fetch targets
  useEffect(() => {
    const fetchTargets = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:8000/targets");
        setTargets(response.data.targets);
        setFilteredTargets(response.data.targets);
      } catch (err) {
        console.error("Error fetching targets:", err);
      }
    };
    fetchTargets();
  }, []);



  // Modify the tweet submission handler to add tweets to the queue
  const handleTweetSubmit = () => {
    if (tweetQueue.length === 0 && !isProcessingQueue) {
      handleTweetAnalysis();
    } else {
      setTweetQueue((prev) => [...prev, tweetUrl]);
    }
  };

  const handleTweetAnalysis = useCallback(async () => {
    if (isRateLimited) {
      setError(`Please wait ${cooldownTimer} seconds before trying again`);
      return;
    }
  
    setLoading(true);
    setResult(null);
    setProbabilities(null);
    setAttentionScores([]);
    setTokens([]);
    setError("");
  
    try {
      const response = await axios.post("http://127.0.0.1:8000/analyze_tweet/", {
        tweet_url: tweetUrl,
        target: target || null,
      });
  
      setResult(response.data.stance);
      setProbabilities(response.data.probabilities);
      setAttentionScores(response.data.attention);
      setTokens(response.data.tokens);
      setHasSubmitted(true);
  
      setHistory((prev) => [
        ...prev,
        {
          statement: response.data.tweet_text,
          target: response.data.target,
          stance: response.data.stance,
          probabilities: response.data.probabilities,
          isTweet: true,
        },
      ]);
  
      // Reset the form after successful analysis
      setTweetUrl("");
      setTarget("");
      setFilteredTargets(targets); // Reset filtered targets to full list
  
    } catch (err) {
      if (err.response?.status === 429) {
        startCooldownTimer();
        setError("Rate limit reached. Please wait before trying again.");
      } else {
        setError(err.response?.data?.detail || "Error analyzing tweet");
      }
    } finally {
      setLoading(false);
    }
  }, [isRateLimited, tweetUrl, target, cooldownTimer, targets]);

    // Process the tweet queue
    useEffect(() => {
      const processQueue = async () => {
        if (tweetQueue.length > 0 && !isProcessingQueue) {
          setIsProcessingQueue(true);
          
          const nextTweet = tweetQueue[0];
          setTweetUrl(nextTweet);
          
          try {
            await handleTweetAnalysis(); // this needs to be in the dependency array
          } catch (error) {
            console.error("Error processing tweet:", error);
          }
          
          setTweetQueue((prev) => prev.slice(1));
          setIsProcessingQueue(false);
          
          // Delay between processing tweets
          await new Promise((resolve) => setTimeout(resolve, 2000));
        }
      };
    
      processQueue();
    }, [tweetQueue, isProcessingQueue, handleTweetAnalysis]); 
    

  const startCooldownTimer = () => {
    setIsRateLimited(true);
    setCooldownTimer(60); // 60 second cooldown
    
    const interval = setInterval(() => {
      setCooldownTimer((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          setIsRateLimited(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  // Filter targets as the user types
  const handleSearchChange = (e) => {
    const searchQuery = e.target.value.toLowerCase();
    setTarget(searchQuery); // Update the target state with the current input
    const filtered = targets.filter((t) =>
      t.toLowerCase().includes(searchQuery)
    );
    setFilteredTargets(filtered);
  };

  // When the user selects a target from the list
  const handleTargetSelect = (target) => {
    setTarget(target);
    setFilteredTargets(targets); // Reset filtered targets after selection
  };

  // Handle form submission for text input
  const handleSubmit = async () => {
    setLoading(true);
    setResult(null);
    setProbabilities(null);
    setAttentionScores([]);
    setTokens([]);
    setError("");

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict/", {
        statement,
        target: target || null
      });

      setResult(response.data.stance);
      setProbabilities(response.data.probabilities);
      setAttentionScores(response.data.attention);
      setTokens(response.data.tokens);
      setHasSubmitted(true);

      setHistory((prev) => [
        ...prev,
        {
          statement,
          target: response.data.target,
          stance: response.data.stance,
          probabilities: response.data.probabilities
        }
      ]);
    } catch (err) {
      setError("Error detecting stance.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div className="min-h-screen bg-gray-100 relative w-full overflow-x-hidden">
      <div className="container mx-auto px-4 py-4 sm:py-8 max-w-7xl">
        <motion.div className="flex flex-col items-center">
          {/* Header Section */}
          <motion.div className="text-center mb-4 sm:mb-8 w-full">
            <h1 className="text-2xl sm:text-4xl font-bold text-gray-800">Political Stance Detection</h1>
            <p className="text-gray-600 mt-2 text-sm sm:text-base">
              Select a topic and enter a statement or analyze a tweet to detect its stance.
            </p>
          </motion.div>

          {/* Mode Selection */}
          <div className="w-full max-w-2xl mb-4">
            <div className="flex rounded-lg overflow-hidden bg-gray-200 p-1">
              <button
                className={`flex-1 py-2 px-4 rounded-lg ${
                  analysisMode === "text" ? "bg-white shadow" : ""
                }`}
                onClick={() => setAnalysisMode("text")}
              >
                Text Input
              </button>
              <button
                className={`flex-1 py-2 px-4 rounded-lg flex items-center justify-center gap-2 ${
                  analysisMode === "tweet" ? "bg-white shadow" : ""
                }`}
                onClick={() => setAnalysisMode("tweet")}
              >
                <Twitter size={20} /> Tweet Analysis
              </button>
            </div>
          </div>

          {/* Input Section */}
          <div className="w-full max-w-2xl bg-white p-4 sm:p-6 rounded-lg shadow-md">
            {error && (
              <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg">
                {error}
              </div>
            )}

            <div className="mb-4">
              <input
                type="text"
                className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Search or type your target"
                value={target}
                onChange={handleSearchChange}
              />
              {filteredTargets.length > 0 && (
                <div className="mt-2 max-h-40 overflow-y-auto border border-gray-300 rounded-lg bg-white">
                  {filteredTargets.map((t) => (
                    <div
                      key={t}
                      className="p-3 hover:bg-gray-200 cursor-pointer"
                      onClick={() => handleTargetSelect(t)}
                    >
                      {t}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {analysisMode === "text" ? (
              <textarea
                className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 mb-4"
                rows={4}
                placeholder="Type your political statement here..."
                value={statement}
                onChange={(e) => setStatement(e.target.value)}
              />
            ) : (
              <input
                type="text"
                className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 mb-4"
                placeholder="Paste Tweet URL here..."
                value={tweetUrl}
                onChange={(e) => setTweetUrl(e.target.value)}
              />
            )}

            <button
              className="w-full bg-blue-500 text-white py-3 px-4 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-blue-300"
              onClick={analysisMode === "text" ? handleSubmit : handleTweetSubmit}
              disabled={loading || (analysisMode === "text" ? !statement.trim() : !tweetUrl.trim())}
            >
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </div>

          {/* Results Section */}
          <AnimatePresence>
            {hasSubmitted && (
              <motion.div
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 50 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                className="mt-4 sm:mt-8"
              >
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-8">
                  {/* Results Section */}
                  <motion.div
                    className="space-y-4 sm:space-y-6"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 0.5 }}
                  >
                    {result && (
                      <motion.div
                        className="bg-white p-4 sm:p-6 rounded-lg shadow-md"
                        initial={{ scale: 0.9 }}
                        animate={{ scale: 1 }}
                        transition={{ duration: 0.3 }}
                      >
                        <h2 className="text-lg sm:text-xl font-bold text-gray-800 mb-4">Result</h2>
                        <p className="text-base sm:text-lg">Detected Stance: <span className="font-bold">{result}</span></p>
                      </motion.div>
                    )}

                    {probabilities && (
                      <motion.div
                        className="bg-white p-4 sm:p-6 rounded-lg shadow-md"
                        initial={{ scale: 0.9 }}
                        animate={{ scale: 1 }}
                        transition={{ duration: 0.3, delay: 0.2 }}
                      >
                        <h2 className="text-lg sm:text-xl font-bold text-gray-800 mb-4">Confidence Scores</h2>
                        <div className="w-full h-64 sm:h-72">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                              data={[
                                { name: "Against", value: probabilities.Against },
                                { name: "Neutral", value: probabilities.Neutral },
                                { name: "Support", value: probabilities.Support },
                              ]}
                              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="name" />
                              <YAxis />
                              <Tooltip />
                              <Legend />
                              <Bar dataKey="value" fill="#8884d8" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </motion.div>
                    )}
                  </motion.div>

                  {/* Attention Scores and History */}
                  <motion.div
                    className="space-y-4 sm:space-y-6"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 0.7 }}
                  >
                    {tokens.length > 0 && attentionScores.length > 0 && (
                    <motion.div
                      className="bg-white p-4 sm:p-6 rounded-lg shadow-md"
                      initial={{ scale: 0.9 }}
                      animate={{ scale: 1 }}
                      transition={{ duration: 0.3 }}
                    >
                      <h2 className="text-lg sm:text-xl font-bold text-gray-800 mb-4">Attention Scores</h2>
                      <div className="flex flex-wrap gap-2">
                        {tokens.map((token, index) => {
                          const score = attentionScores[index];
                          return (
                            <motion.div
                              key={index}
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ duration: 0.3, delay: index * 0.05 }}
                              className="relative group cursor-pointer"
                            >
                              <div 
                                className="p-2 rounded text-sm sm:text-base transition-colors duration-200"
                                style={{
                                  backgroundColor: `rgba(59, 130, 246, ${score})`,
                                  color: score > 0.5 ? 'white' : 'black'
                                }}
                              >
                                {token}
                              </div>
                              {/* Tooltip */}
                              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap">
                                Score: {(score * 100).toFixed(1)}%
                              </div>
                            </motion.div>
                          );
                        })}
                      </div>
                      
                      {/* Legend */}
                      <div className="mt-4 flex items-center gap-2 text-sm text-gray-600">
                        <div className="flex items-center gap-1">
                          <div className="w-4 h-4 bg-blue-500 bg-opacity-20 rounded"></div>
                          <span>Low Attention</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <div className="w-4 h-4 bg-blue-500 rounded"></div>
                          <span>High Attention</span>
                        </div>
                      </div>
                    </motion.div>
                  )}

                    {history.length > 0 && (
                      <motion.div
                        className="bg-white p-4 sm:p-6 rounded-lg shadow-md"
                        initial={{ scale: 0.9 }}
                        animate={{ scale: 1 }}
                        transition={{ duration: 0.3, delay: 0.4 }}
                      >
                        <h2 className="text-lg sm:text-xl font-bold text-gray-800 mb-4">History</h2>
                        <div className="space-y-3">
                          {history.map((item, index) => (
                            <motion.div
                              key={index}
                              initial={{ opacity: 0, y: 20 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ duration: 0.3, delay: index * 0.1 }}
                              className="p-3 bg-gray-50 rounded-lg"
                            >
                              <p className="font-medium text-sm sm:text-base">{item.statement}</p>
                              <p className="text-xs sm:text-sm text-gray-600 mt-1">
                                Topic: <span className="font-semibold">{item.target}</span> | 
                                Stance: <span className="font-semibold">{item.stance}</span>
                              </p>
                            </motion.div>
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </motion.div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </motion.div>
  );
}
