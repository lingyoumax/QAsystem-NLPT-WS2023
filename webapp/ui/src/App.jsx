import {
    RouterProvider,
    createBrowserRouter,
} from "react-router-dom";
import ChatBox from "./componets/ChatBox.jsx";

function App() {
    const router = createBrowserRouter([
        {
            path: "/",
            element: <ChatBox/>
        }
        ]
    );

    return (
        <RouterProvider router={router} />
    );
}

export default App;
