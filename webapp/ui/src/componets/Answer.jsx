import React, {useState, useRef, useEffect, Fragment} from 'react';
import styles from './Answer.module.css';

function Answer({ msg, reference }) {
    const [isHovered, setIsHovered] = useState(false);
    const ref = useRef(null);
    const handleMouseEnter = () => {
        setIsHovered(true);
    };

    const handleMouseLeave = () => {
        setIsHovered(false);
    };

    return (
        <div>
        <div className={styles.leftbox}  onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave} ref={ref}>
            {msg.content}
        </div>
            {isHovered && reference && (
                <div className={styles.reference} >
                    {reference}
                </div>
            )}
        </div>
    );
};

export default Answer;
