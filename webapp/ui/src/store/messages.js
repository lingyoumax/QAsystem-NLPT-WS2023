import create from "zustand";

const message = create((set) => ({
    messages: [

    ],
    appendMessages: (data) => set((state) => ({ mySavedCV: state.messages.push(data) })),
}));
export default message;
