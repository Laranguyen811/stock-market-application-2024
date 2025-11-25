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
        const response = await fetch('http://localhost:8000/api/simplify') // await the response from the fetch request. await keyword is permitted within the function body, enabling asynchronous, promise-based behaviour to be written in a cleaner style
            const data = await response.json(); // await the json data from the response
            setResult(data); // set the result to the data from the response

        }
    }

}