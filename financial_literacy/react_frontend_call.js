import response from "../.venv/Lib/site-packages/dash/dash-renderer/build/dash_renderer.dev";

const simplifyText = async (text) => {
    const response = await fetch('http://localhost:8000/api/simplify/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({'text': text})
    });
    return response.json();
};

export  { simplifyText };