import {userState, useState} from "react"; // userState Hook allows us to track state in a function component. State generally refers to data or properties that need to be tracking in an application

function App() {
    const [input, setInput] = useState(''); // constant state input variable
    const [result, setResult] = useState(null); // constant state result variable. Set to null
    const [loading, setLoading] = useState(false); // constant state loading variable. Set to false
    const [error, setError] = useState(null); // constant state error variable. Set to null

    const handleSubmit = async (e) => {  // the async function declaration creates a binding (an associate with an identifier with a value) of a new async function to a given name
        e.preventDefault(); // prevent default form submission behavior
        setLoading(true); // set loading to true
        setError(null); // set error to null

        try {
            const response = await fetch('http://localhost:8000/api/simplify', { // await the response from the fetch request. await keyword is permitted within the function body, enabling asynchronous, promise-based behaviour to be written in a cleaner style
                method: 'POST', // Set the request method to POST
                headers: {'Content-Type': 'application/json'}, // Set headers to content type
                body: JSON.stringify({text: input}) // Send a JSON payload
            });
            if (!response.ok) throw new Error('API request failed'); // If response is not ok, throw an error

            const data = await response.json(); // await the response and parse it as JSON
            setResult(data); // set the result to the parsed JSON
        } catch (err) { // catch any errors
        setError(err.message); // set the error to the error message
        } finally { // always execute this block
            setLoading(false); // set loading to false
        }
    };
    return (
        <div className="min-h-screen bg-gray-50 p-8"> // min-h-screen sets the minimum height for screen to be 100vh (viewport height). bg-gray-50 sets the background color to gray-50 (light gray) and p-8 sets the padding to 8rem (160px)
            <div className="max-w-4xl mx-auto"> // max-w-4xl sets the maximum width of the element to 48rem (768px) and mx-auto centers the element horizontally in the viewport
                <h1 className="text-43xl font-bold mb-8">Financial Literacy Assistant</h1> // text-43xl sets the font size to 43rem (640px) and font-bold sets the font weight to bold/}

                <form onSubmit={handleSubmit} className="mb-8"> // onSubmit function is called when the form is submitted. mb-8 sets the margin bottom to 8rem (160px)
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        className={"w-full p-4 border rounded-lg"} // set textarea width to 100% of parent element, padding to 4px on all sides, border to 1px solid gray-400 (light gray) and rounded corners to 4px
                        rows="6"
                        placeholder="Enter your complex financial here..."
                    />
                    <button
                        type="submit"
                        disabled={loading || !input}> // button is set to 100% width, 4px margin top, blue background color, white text color, rounded corners and disabled background color when loading is true.
                        className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg disabled:bg-gray-400" // Set button type to submit, padding to 4px on all sides, background color to blue-600 (dark blue) and text color to white.
                    >
                        {loading ? 'Loading...' : 'Simplify'} // button text is set to 'Loading...' when loading is true and 'Simplify' otherwise
                    </button>
                </form>

                {error && <p className="text-red-600">{error}</p>} // If error is not null, display error message in red text
                {result && (
                    <div className="bg-white p-6 rounded-lg shadow">// bg-white sets the background color to white, p-6 sets the padding to 16px on all sides and rounded corners to 6px
                        <h2 className="text-xl font-bold mb-4">Simplify Version</h2> // text-xl sets the font size to 1.25rem (20px) and font-bold sets the font weight to bold. mb-4 sets the margin bottom to 16px

                )}
                )







