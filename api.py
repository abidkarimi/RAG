from flask import Flask, request, jsonify
from flask_cors import CORS

# from JurisAiService import process_data,callerFun
from assistantOpenAPI import get_assistant_response
# from openAiApi import callerFun
# from my_streamlit_app import main, main1
# from chatbot_app import check, llm_pipeline_using_deepset

app = Flask(__name__)
CORS(app)

@app.route('/api', methods=['GET'])
def get_data():
    # You can customize this function to accept parameters
    # from the request and return data accordingly
    # main()  # Run your Streamlit app
    # Here, you would collect data from your Streamlit app
    # and return it as JSON
    input_param = request.args.get('input')
    google_places_api_key = request.args.get('key')
    types_param = request.args.get('types')
    print("Received params:", input_param)
    # data = {'message': llm_pipeline_using_deepset(input_param, "")}
    return jsonify(
        get_assistant_response(input_param)
        # process_data()
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
