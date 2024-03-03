import React, {Fragment, useEffect, useRef, useState} from "react";
import styles from './ChatBox.module.css'
import message from "../store/messages.js";
import MyQuestion from "./MyQuestion.jsx";
import Answer from "./Answer.jsx";
import Progress from "./Progress.jsx";
import {postQuestionAPI} from "../../lib/postQuestion.js";
import {fetchAnswerAPI, fetchStatusAPI} from "../../lib/fetchStatusAPI.js";
function ChatBox(){
    const boxRef = useRef(null);
    const [inputValue, setInputValue] = useState('');
    const [addLine, setAddLine] = useState(false);
    const [inputStatus, setInputStatus] = useState(false);
    const [requestStatus, setRequestStatus] = useState(false);
    const [latest_question, setLatest_question] = useState();
    const [start_year, setStart_year] = useState("");
    const [end_year, setEnd_year] = useState("");
    const [author, setAuthor] = useState("");
    const [reference, setReference] = useState(null);
    const [status, setStatus] = useState({
        input: false,
        retrieval: false,
        answerGenerating: false,
        output: false,
    });
    const [processing, setProcessing] = useState({
        input: false,
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
    const handleKeyDown = async (e) => {
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
            setInputValue('');
            setProcessing({
                input: true,
                retrieval: false,
                answerGenerating: false,
                output: false,
            })
            try {
                const data = await postQuestionAPI(inputValue.trim(), temp_stamp, start_year+"-"+end_year, author); // 假定这是你的 API 请求函数
                if (data.message === 'question input successful') {
                    setProcessing({
                        input: false,
                        retrieval: true,
                        answerGenerating: false,
                        output: false,
                    })
                    setStatus({
                        input: true,
                        retrieval: true,
                        answerGenerating: false,
                        output: false,
                    })

                    setRequestStatus(true)
                }
            } catch (error) {
                console.error('数据获取失败：', error);
            }

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
                        if(data.status.retrieval){
                            setReference(data.status.reference)
                            setProcessing({
                                input: false,
                                retrieval: false,
                                answerGenerating: true,
                                output: false,
                            })
                        }
                        if(data.status.answerGenerating){
                            setProcessing({
                                input: false,
                                retrieval: false,
                                answerGenerating: false,
                                output: true,
                            })
                        }

                        setStatus(data.status);
                        if (data.status.output) {
                            setRequestStatus(false); // 如果有输出，则不再需要定期检查
                            clearInterval(intervalId);
                        }
                    }
                } catch (error) {
                    console.error('数据获取失败：', error);
                }
            }, 15000);
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
                // console.log(data)
                if (data.res === 'success') {
                    setInputStatus(false)
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
        <div className={styles.filter}>
            <div className={styles.pubdate}>
                <label>Pubdate:</label>
                    <input placeholder="from" onChange={(e)=>{setStart_year(e.target.value)}}/>-
                    <input placeholder="to" onChange={(e)=>{setEnd_year(e.target.value)}}/>
            </div>
            <div className={styles.author}>
                <label>Author:</label>
                <input placeholder="author name" onChange={(e)=>{setAuthor(e.target.value)}}/>
            </div>
        </div>

        <div className={styles.wrapper}>
            <div className={styles.chatbox}>
                <div className={styles.box} ref={boxRef}>
                    {messages.map((msg, index) => {
                        if(msg.type === 'question'){
                            return <MyQuestion key={index} msg={msg}/>
                        }else{
                            return  <Answer  key={index} msg={msg} reference={reference}/>
                        }
                    })}
                </div>
                <textarea
                    className={styles.input_question}
                    value={inputValue}
                    disabled={inputStatus}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    placeholder="Type a question..."
                />
            </div>



            <div className={styles.progress}>
                <Progress content="Question Input" status={status.input} processing={processing.input}/>
                <Progress content="Retrieval" status={status.retrieval} processing={processing.retrieval}/>
                <Progress content="Answer Generation" status={status.answerGenerating} processing={processing.answerGenerating}/>
                <Progress content="Finished" status={status.output} processing={processing.output}/>
            </div>
        </div>
    </Fragment>
}
export default ChatBox
