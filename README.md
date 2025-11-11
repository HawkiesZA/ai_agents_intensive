# AI Agents Intensive

This project is a collection of examples for the [Agent Development Kit](https://github.com/google/agent-development-kit) (ADK), as done through the 5 day agents intensive course from [Kaggle](https://www.kaggle.com/learn-guide/5-day-agents).

## Setup

### uv
This project uses uv to manage dependencies. To install dependencies, run:

```bash
uv install
```

### Gemini API Key
You will need to set the `GOOGLE_API_KEY` environment variable to your Gemini API key.

You can get an API key from [here](https://aistudio.google.com/app/api-keys).

Then create a .env file in the root directory of the project and add the following line:

```bash
GOOGLE_API_KEY=your_api_key
```

## Running the app
### Running the app
To run an app, run:

```bash
uv run <day_x>/<app_name>.py
```

### Running the web server
You will need to create a .env file in the `sample-agent` directory and add the following line:

```bash
GOOGLE_API_KEY=your_api_key
```

To run the web server, run:

```bash
uv run adk web
```
