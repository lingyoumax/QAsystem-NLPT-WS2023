export async function postQuestionAPI(question, temp_stamp, year, author){
    try {
        var myHeaders = new Headers();
        myHeaders.append("Content-Type", "application/json");

        const response = await fetch("http://localhost:8000/sendQuestion", {
            // 更新为你的注册路由
            method: "POST",
            headers: myHeaders,
            body: JSON.stringify({
                question, time_stamp: temp_stamp, year, author
            }),
        });
        return await response.json();
    } catch (error) {
        console.error("Error during signup:", error);
    }
}