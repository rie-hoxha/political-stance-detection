# Project Title: Political Stance Detection
## Tagline
A comprehensive application for analyzing political statements and tweets.

## Overview
This project is a Political Stance Detection application that allows users to analyze tweets and detect the stance of political statements. It integrates with the Twitter API to fetch tweets and uses machine learning models to predict stances based on the content of the statements.

## Backend Documentation
### FastAPI Setup
The backend is built using FastAPI. The following endpoints are available:

- **`/analyze_tweet/`**: 
  - **Method**: POST
  - **Description**: Analyzes a tweet based on its URL and returns the detected stance.
  - **Request Body**: 
    ```json
    {
      "tweet_url": "https://twitter.com/user/status/tweet_id",
      "target": "optional_target"
    }
    ```
  - **Response**: Returns a JSON object containing the stance and other relevant data.

- **`/predict/`**: 
  - **Method**: POST
  - **Description**: Predicts the stance of a user-provided statement.
  - **Request Body**: 
    ```json
    {
      "statement": "Your political statement here",
      "target": "optional_target"
    }
    ```
  - **Response**: Returns a JSON object with the predicted stance and confidence scores.

### Twitter API Integration
To use the Twitter API, you need to create a `.env` file in the root directory of the project with the following keys:

```
TWITTER_BEARER_TOKEN=your_bearer_token
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_SECRET=your_access_secret
```

These keys are essential for authenticating requests to the Twitter API. Make sure to keep this file secure and do not share it publicly.

## Frontend 
The frontend is built using React and provides an intuitive user interface for analyzing tweets and statements. Users can input either a tweet URL or a text statement for analysis.

### User Interface
- **Text Input Mode**: Users can type a political statement to analyze. This mode is useful for users who want to evaluate their own statements.
- **Tweet Analysis Mode**: Users can paste a tweet URL to analyze its stance. This mode fetches the tweet's content and provides an analysis based on the text.

### Features
- **Fetching Targets**: The application fetches available targets from the backend, allowing users to select specific topics for analysis.
- **Displaying Results**: After analysis, the application displays the detected stance along with confidence scores in a user-friendly format.
- **Rate Limiting**: The frontend handles rate limiting and provides user feedback through loading states and error messages.

### Component Breakdown
- **State Management**: The application uses React's `useState` and `useEffect` hooks to manage state and side effects.
- **Axios for API Calls**: Axios is used to make HTTP requests to the backend for fetching targets and analyzing tweets.
- **Responsive Design**: The application is designed to be responsive, ensuring a good user experience across different devices.

## Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/rie-hoxha/political-stance-detection.git
   cd political-stance-detection
   ```
2. Install the necessary dependencies for the backend by running:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your Twitter API keys as described above.


4. Navigate to the frontend directory and install the frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

## Usage Instructions
1. Start the FastAPI server by running:
   ```bash
   uvicorn backend.app:app --reload
   ```
   This command will start the server in development mode, allowing for hot reloading of changes.
   
2. Open the frontend application in your browser. You can do this by navigating to `http://localhost:3000` or the specified port in your React app.

3. Use the interface to analyze tweets or text statements. Simply enter the required information and click the "Analyze" button to see the results.
