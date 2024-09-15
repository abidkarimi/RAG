from flask import Flask, jsonify, request
from assistantOpenAPI import get_assistant_response
# Initialize the Flask app
app = Flask(__name__)

# Define a route for the GET API endpoint
@app.route('/open-ai', methods=['GET'])
def greet():
    # Get the 'name' parameter from the query string
    query = request.args.get('query', 'World')  # Default to 'World' if no name is provided
    # Create a greeting message
    message = f"Hello, {query}!"
    # Return the message as a JSON response
    return jsonify({"message": get_assistant_response(query)})

# Main entry point
if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
