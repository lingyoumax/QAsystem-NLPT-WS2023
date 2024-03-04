# How to Use

## Preparing the Files
To get started with the web app, you need to prepare the necessary files. There are two methods to do this:

### Method 1: Download ZIP File
1. Navigate to [the link](https://drive.google.com/file/d/1ZTs8CF1LbVpMUhTraYlCxwvl6ifSjbOm/view?usp=drive_link) to download the web app as a ZIP file.
2. Extract the ZIP file to your desired location to begin setup.

### Method 2: Clone GitHub Repository and Download Files
1. Clone the GitHub project to your local machine
2. After cloning the project, download the required files from the three provided links.[1](https://drive.google.com/file/d/1BXSFNRFe_usZAhrJVpX2f-KkdsqvGtfX/view?usp=drive_link) [2](https://drive.google.com/file/d/1eTuG85UPLnzYclz_QWaHhu6RyiQrxOFI/view?usp=drive_link) [3](https://drive.google.com/file/d/1QXCMTbAqbG4kYhL0nsjPGPyvj0rzGfWe/view?usp=drive_link)
3. Place the downloaded files into the `/webapp/backend/retrieval` directory within the cloned project folder.

## Running the Web App
After preparing the files, follow these steps to run the web app:
1. Open a terminal or command prompt.
2. Change directory to the web app's folder by running the following command:  
   `cd /path/to/webapp`  
   Replace `/path/to/webapp` with the actual path to your web app directory.
3. Start the web app using Docker Compose with the following command:  
   `docker compose up`  
   This command will set up everything needed for the web app to run. The terminal will display a URL, typically `http://localhost:5173/`, once the setup is complete and the web app is running.
4. Open a web browser and navigate to the displayed URL to start using the web app.

## Important Notes
Before you start using the web app, please ensure you are aware of the following requirements:

1. **Docker Required:** The setup and execution of this web app rely on Docker. Make sure you have Docker installed on your system. If you don't have Docker installed, visit [Docker's official website](https://www.docker.com/) for installation instructions.
2. **GPU Required:** The Question Answering (QA) model used in this project requires significant computational resources. Due to the original model's resource-intensive nature, model quantization is employed to optimize performance. [The bitsandbytes library](https://github.com/TimDettmers/bitsandbytes), which is used for quantization, requires a GPU. Ensure your system is equipped with a compatible GPU to run the project successfully.
3. **Initial Model Download:** Upon initial startup, the web app will download a large model file, approximately 16GB in size. This initial setup process can be quite time-consuming, depending on your internet connection speed. Ensure you have a stable and fast internet connection to minimize the setup time.

# Project Structure

This file outlines the structure of a web application that utilizes FastAPI as the backend framework and React as the frontend framework. The application is designed to handle question submissions, track their processing status, and retrieve answers. Below is a detailed breakdown of the project's components and API endpoints.

## Backend Framework: FastAPI

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints. It's known for its speed, ease of use, and ability to automatically generate interactive API documentation.

## Frontend Framework: React

React is a declarative, efficient, and flexible JavaScript library for building user interfaces. It enables the creation of complex UIs from small, isolated pieces of components.

## API

The backend provides a RESTful API with endpoints to submit questions, check their processing status, and retrieve answers. Each endpoint is designed with error handling to ensure robustness and reliability.

### Endpoints

#### 1. POST /sendQuestion

Submits a question along with related information like text, timestamp, year range, and author. Successful submissions initiate background processing.

- **Request Body:**
  - `question`: Text of the question.
  - `time_stamp`: Unique identifier for submission time.
  - `year`: Relevant year range or "-" if unspecified.
  - `author`: Name of the submitter.
- **Response:**
  - Success: JSON object with a success message.
  - Error: HTTP 500 with error details.

#### 2. POST /questionStatus

Checks the processing status of a question using its timestamp, detailing each processing stage.

- **Request Body:**
  - `TIME_STAMP`: Timestamp of the question.
- **Response:**
  - Success: JSON object with status details.
  - Error: HTTP 500 for server issues or HTTP 404 if the question is not found.

#### 3. POST /getAnswer

Retrieves the answer to a processed question using its timestamp.

- **Request Body:**
  - `TIME_STAMP`: Timestamp of the question.
- **Response:**
  - Success: JSON object with the answer.
  - Error: HTTP 500 for server issues or HTTP 404 if the answer is not available.

## Error Handling

The API employs comprehensive error handling to manage and communicate issues effectively. Each endpoint returns an HTTP error response with a relevant status code (404 for "Not Found", 500 for "Internal Server Error") and a detailed error message in case of exceptions. This approach ensures that clients receive clear and actionable feedback on any failures.