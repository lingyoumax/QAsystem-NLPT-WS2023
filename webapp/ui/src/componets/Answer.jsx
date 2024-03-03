import React, { useState, useRef } from 'react';
import styles from './Answer.module.css';

function Answer({ msg, reference }) {
    // reference = "vcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwejvcnewnkjewdnfjkwejkfnwej"
    // console.log(reference)
    const [hoverBoxVisible, setHoverBoxVisible] = useState(false);
    const [hoverBoxPosition, setHoverBoxPosition] = useState({ x: 0, y: 0 });
    const leftBoxRef = useRef(null);

    const handleMouseEnter = () => {
        setHoverBoxVisible(true);
    };

    const handleMouseLeave = () => {
        setHoverBoxVisible(false);
    };

    const handleMouseMove = (e) => {
        if (!leftBoxRef.current) return;

        const hoverBoxWidth = 100; // 你可以根据实际情况调整 hoverBox 的宽度
        const leftBoxRect = leftBoxRef.current.getBoundingClientRect();
        let x = e.clientX - leftBoxRect.left; // 鼠标相对于 leftBox 的 X 坐标
        let y =  e.clientY - leftBoxRect.top; // 鼠标相对于 leftBox 的 Y 坐标

        // 确保 hoverBox 不会超出 leftBox 的边界
        // x = Math.min(x, leftBoxRect.width + hoverBoxWidth);
        // y = Math.max(0, y);

        setHoverBoxPosition({ x, y });
    };

    return (
        <div
            className={styles.leftbox}
            ref={leftBoxRef}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onMouseMove={handleMouseMove}
        >
            {msg.content}
            {hoverBoxVisible && (
                <div
                    className={styles.hoverBox}
                    style={{
                        position: 'absolute',
                        left: `${hoverBoxPosition.x}px`,
                        top: `${hoverBoxPosition.y}px`,
                    }}
                >
                    {reference}
                </div>
            )}
        </div>
    );
}

export default Answer;
