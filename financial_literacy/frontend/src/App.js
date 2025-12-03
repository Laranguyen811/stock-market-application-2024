import {userState, useState} from "react"; /* userState Hook allows us to track state in a function component. State generally refers to data or properties that need to be tracking in an application */

function App() {
    const [input, setInput] = useState(''); /* constant state input variable */
    const [result, setResult] = useState(null); /* constant state result variable. Set to null */
    const [loading, setLoading] = useState(false); /* constant state loading variable. Set to false */
    const [error, setError] = useState(null); /* constant state error variable. Set to null */

    const handleSubmit = async (e) => {  /* the async function declaration creates a binding (an associate with an identifier with a value) of a new async function to a given name */
        e.preventDefault(); /* prevent default form submission behavior */
        setLoading(true); /* set loading to true */
        setError(null); /* set error to null */

        try {
            const response = await fetch('http://localhost:8000/api/simplify', { /* await the response from the fetch request. await keyword is permitted within the function body, enabling asynchronous, promise-based behaviour to be written in a cleaner style */
                method: 'POST', /* Set the request method to POST */
                headers: {'Content-Type': 'application/json'}, /* Set headers to content type */
                body: JSON.stringify({text: input}) /* Send a JSON payload */
            });
            if (!response.ok) throw new Error('API request failed'); /* If response is not ok, throw an error */

            const data = await response.json(); /* await the response and parse it as JSON */
            setResult(data); /* set the result to the parsed JSON */
        } catch (err) { /* catch any errors */
        setError(err.message); /* set the error to the error message */
        } finally { /* always execute this block */
            setLoading(false); /* set loading to false */
        }
    };
    return (
        /* min-h-screen sets the maximum height . bg-gradient-to-br sets the background gradient to blue-50 to emerald-50 to teal-50 */
        <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50">
            <div className="max-w-5xl mx-auto px-6 py-12">
                {/* Header */}
                {/* text-center sets the text alignment to center, mb-12 sets the bottom margin to 12px */}
                <div className="text-center mb-12"
                    {/* inline-block sets the display property to inline-block, p-3 sets the padding to 16px on all sides and rounded corners to 3px, bg-green-100 sets the background color to green-100 (light green) and mb-4 sets the bottom margin to 4px */}
                    <div className="inline-block p-3 bg-green-100 rounded-full mb-4">Financial Literacy Assistant</div>
                    {/* w-12 h-12 sets the width and height to 12px, text-green-600 sets the text color to green-600 (dark green), fill sets to none, stroke is the current color, viewBox sets the viewbox to 0 0 24 24 */}
                        <svg className="w-12 h-12 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            {/* strokeLineCap (a presentation attribute defining the shape to be used at the end of open subpaths when they are stroked) is set to round,  (attribute is a presentation attribute defining the shape to be used at the corners of paths when they are stroked). Stroke width is 2. Has book icon the your header */}
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />

                        </svg>
                </div>
            {/* text-5xl - Font size (48px), font-bold - Bold weight, bg-gradient-to-r - Gradient flows left to right, from-green-600 - Gradient starts at green, to-emerald-600 - Gradient ends at emerald, bg-clip-text - Clips background to text shape, text-transparent - Makes text transparent so gradient shows through, mb-3 - Margin bottom (0.75rem spacing) */}
            <h1 className="text-5xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent mb-3">
            Financial Literacy Assistant
          </h1>
            {/* textgray-600 text-lg - Font size (16px), mb-8 - Margin bottom (2rem spacing) */}/}
        </div>
        {/*Input Card */}
        {/* bg-white rounded-2xl shadow-xl p-8 mb-8 border border-gray-100 - Sets the background color to white, rounded corners to 2xl, shadow to 8px, border to 1px solid gray-100 */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8 border border-gray-100">
            {/* bg-gradient-to-r from-green-600 to-emerald-600 p-6 - Sets the background gradient to blue-50 to emerald-50 to teal-50 */}
            <div className="bg-gradient-to-r from-green-600 to-emerald-600 p-6">
                {/* */}
                <h2 className="text-2xl font-bold text-white flex items-center">
export default App;







