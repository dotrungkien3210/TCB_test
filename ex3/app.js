const express = require('express');
const app = express();
app.use(express.json());
const PORT=5000;

app.use(
    express.urlencoded({
      extended: true,
    })
  );
const itemsRouter = require('./router/item');
app.use('/items', itemsRouter);

app.listen(PORT,()=>console.log(`listening on port ${PORT}`))