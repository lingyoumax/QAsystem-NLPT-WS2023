import styles from './MyQuestion.module.css'
function MyQuestion({msg}){
    return <div className={styles.rightbox}>
        {msg.content}
    </div>
}
export default MyQuestion