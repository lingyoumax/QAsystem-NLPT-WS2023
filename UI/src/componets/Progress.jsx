import styles from './Progress.module.css'
function Progress({content, status}){
    return <div className={status ? styles.finish: styles.notfinish}>
        {content}
    </div>
}
export default Progress