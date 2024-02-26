import create from "zustand";

const message = create((set) => ({
    messages: [
        {
            type: 'question',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        },
        {
            type: 'answer',
            time_stamp: new Date(),
            content: 'hello from the world ',
        }
    ],
    appendMessages: (data) => set((state) => ({ mySavedCV: state.messages.push(data) })),
}));
export default message;
