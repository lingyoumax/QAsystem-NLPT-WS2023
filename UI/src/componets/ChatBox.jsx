import React, {Fragment, useEffect, useRef, useState} from "react";
import styles from './ChatBox.module.css'
import message from "../store/messages.js";
import MyQuestion from "./MyQuestion.jsx";
import Answer from "./Answer.jsx";
import Progress from "./Progress.jsx";
function ChatBox(){
    const boxRef = useRef(null);
    const [inputValue, setInputValue] = useState('');
    const [addLine, setAddLine] = useState(false);
    const [inputStatus, setInputStatus] = useState(false);
    const [requestStatus, setRequestStatus] = useState(false);
    const [latest_question, setLatest_question] = useState();
    const [status, setStatus] = useState({
        input: true,
        classification: false,
        retrieval: false,
        answerGenerating: false,
        output: false,
    });

    const {messages, appendMessages} = message(
        (state)=>({
            messages:state.messages,
            appendMessages:state.appendMessages,
        })
    )
    const handleInputChange = (e) => {
        setInputValue(e.target.value);
    };
    const handleKeyDown = (e) => {
        if (e.key === 'Enter') { // 检查是否按下了回车键
            let temp_stamp = new Date()
            setLatest_question(temp_stamp)
            appendMessages({
                type: 'question',
                time_stamp: temp_stamp,
                content: inputValue.trim(),
            })
            setAddLine(!addLine)
            setInputStatus(true)
            setRequestStatus(true)

            // send post restful API to backend
            setInputValue('');
        }
    };

    useEffect(() => {
        let intervalId;
        if (requestStatus) {
            intervalId = setInterval(async () => {
                try {
                    const data = await fetchStatusAPI(latest_question); // 假定这是你的 API 请求函数
                    if (data.res === 'success') {
                        setStatus(data.status);
                        if (data.status.output) {
                            setRequestStatus(false); // 如果有输出，则不再需要定期检查
                            clearInterval(intervalId);
                        }
                    }
                } catch (error) {
                    console.error('数据获取失败：', error);
                }
            }, 1000);
        }

        return () => {
            if (intervalId) {
                clearInterval(intervalId);
            }
        };
    }, [requestStatus]);

    useEffect(() => {
        async function fetchAnswer() {
            try {
                const data = await fetchAnswerAPI(latest_question); // 假定这是另一个 API 请求函数
                if (data.res === 'success') {
                    appendMessages({
                        type: 'answer',
                        time_stamp: latest_question,
                        content: data.answer.trim(),
                    })
                    setAddLine(!addLine)
                }
            } catch (error) {
                console.error('获取答案失败：', error);
            }
        }

        if (status.output) {
            fetchAnswer();
        }
    }, [status.output]); // 当 status.output 变化时触发

    useEffect(() => {
        if (boxRef.current) {
            boxRef.current.scrollTop = boxRef.current.scrollHeight; // 滚动到底部
        }
    }, [addLine]);

    return <Fragment>
        <div className={styles.wrapper}>
            <div className={styles.chatbox}>
                <div className={styles.box} ref={boxRef}>
                    {messages.map((msg, index) => {
                        if(msg.type === 'question'){
                            return <MyQuestion key={index} msg={msg}/>
                        }else{
                            return  <Answer  key={index} msg={msg}/>
                        }
                    })}
                </div>
                <input
                    className={styles.input_question}
                    value={inputValue}
                    disabled={inputStatus}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    placeholder="Type a question..."
                />
            </div>
            <div className={styles.progress}>
                <Progress content="Question Input" status={status.input}/>
                <Progress content="Question Classification" status={status.classification}/>
                <Progress content="Retrieval" status={status.retrieval}/>
                <Progress content="Answer Generation" status={status.answerGenerating}/>
                <Progress content="Finished" status={status.output}/>
            </div>
        </div>
    </Fragment>
}
export default ChatBox
