const handleApiError = (error) => {
  if (error.response) {
    // The request was made and the server responded with a status code
    // that falls out of the range of 2xx
    const status = error.response.status;
    const data = error.response.data;

    switch (status) {
      case 400:
        return `Bad Request: ${data.detail || 'Invalid input'}`;
      case 401:
        return 'Unauthorized: Please log in again';
      case 403:
        return 'Forbidden: You do not have permission to perform this action';
      case 404:
        return 'Not Found: The requested resource could not be found';
      case 500:
        return 'Internal Server Error: Please try again later';
      default:
        return `An error occurred: ${data.detail || 'Please try again'}`;
    }
  } else if (error.request) {
    // The request was made but no response was received
    return 'Network Error: Unable to connect to the server';
  } else {
    // Something happened in setting up the request that triggered an Error
    return `An unexpected error occurred: ${error.message}`;
  }
};

export default handleApiError;