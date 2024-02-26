import styles from './Answer.module.css'
function Answer({msg}){
    return <div className={styles.leftbox}>
        {msg.content}
    </div>
}
export default Answer