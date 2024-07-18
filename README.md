
# AI Chat Interface with Streaming and Search Capabilities

## Overview

This project is a web application that integrates an AI chat interface with advanced search capabilities and streaming text generation. The application utilizes various technologies including Django, Flask, OpenAI, Hugging Face, and ChromaDB to deliver an interactive and dynamic user experience.

## Features

- **Interactive AI Chat Interface**: A responsive chat interface that supports both regular conversations and code snippets.
- **Streaming Text Generation**: Displays chat responses in a streaming manner to enhance user experience.
- **Advanced Search**: Integration with ChromaDB for powerful search capabilities, including parsing and displaying results with dynamic headers.
- **Data Management**: Handles search results and displays them in a structured format with headings and bullet points.

## Technologies Used

- **Backend**:
  - **Django**: Framework for building the web application's server-side logic.
  - **Hugging Face API**: Utilized for additional AI models and capabilities.
  - **FAISSDB**: For advanced search and data retrieval functionalities.

- **Frontend**:
  - **HTML/CSS**: For structuring and styling the web pages.
  - **JavaScript**: For implementing dynamic content updates and streaming effects.

## Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/21bq1a4232/Chat-with-pdf.git
cd Chatpdf
```

### Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configuration

1. **API Keys**: Set up your API keys for OpenAI and Hugging Face. You may need to create a `.env` file or configure environment variables:

   ```settings.py

   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```

2. **ChromaDB Setup**: Ensure that ChromaDB is correctly configured for search functionality.

### Running the Application

- **Django Server**: Start the Django development server.
  ```bash
  python manage.py runserver
  ```

### Accessing the Application

- Open your web browser and navigate to `http://127.0.0.1:8000` for the Django application.
- For the Flask chat interface, access it via the URL specified in the Flask server configuration.

## Usage

1. **Search Functionality**: Use the search form to query ChromaDB. Results will be displayed with dynamic headers and bullet points.
2. **Chat Interface**: Interact with the AI chatbot, which supports both code and text responses, with a streaming display of results.

## Contribution

Contributions are welcome! Please submit issues, pull requests, or improvements to enhance the functionality and user experience of the application.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or support, please contact:

- **Email**: deeplearning200416@gmail.com
- **GitHub**: [Your GitHub Profile](https://github.com/21bq1a4232)

