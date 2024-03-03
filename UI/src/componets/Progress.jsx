import styles from './Progress.module.css'
function Progress({content, status, processing}){

    if(processing)
        return <div className={styles.processing}>
            {content}
        </div>

    return <div className={status ? styles.finish: styles.notfinish}>
        {content}
    </div>
}
export default Progress