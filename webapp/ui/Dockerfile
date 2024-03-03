From node

WORKDIR /app

COPY ./vite.config.js ./
COPY ./package*.json ./

RUN npm install

USER root
COPY ./ ./

EXPOSE 5173

ENV HOST=0.0.0.0

CMD npm run dev