export async function fetchStatusAPI(TIME_STAMP){
    console.log('fetchStatusAPI', TIME_STAMP);
    try {
        var myHeaders = new Headers();
        myHeaders.append("Content-Type", "application/json");

        const response = await fetch(`http://localhost:8000/questionStatus`, {
            // 更新为你的注册路由
            method: "POST",
            headers: myHeaders,
            body: JSON.stringify({TIME_STAMP})
        });
        return await response.json();
    } catch (error) {
        console.error("Error during signup:", error);
    }
}


export async function fetchAnswerAPI(TIME_STAMP){
    console.log('fetchAnswerAPI', TIME_STAMP);
    try {
        var myHeaders = new Headers();
        myHeaders.append("Content-Type", "application/json");

        const response = await fetch(`http://localhost:8000/getAnswer`, {
            // 更新为你的注册路由
            method: "POST",
            headers: myHeaders,
            body: JSON.stringify({TIME_STAMP})
        });
        return await response.json();
    } catch (error) {
        console.error("Error during signup:", error);
    }
}