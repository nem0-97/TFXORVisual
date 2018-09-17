let model;

//training data
let trIn=tf.tensor2d([[0,0],[0,1],[1,0],[1,1]]);
let trOut=tf.tensor2d([[0],[1],[1],[0]]);

//input data for model
let inputs=[];

let res=40;
let cols;
let rows;

function setup(){
  createCanvas(innerWidth,innerHeight);
  background(0);

  //create sequential model
  model=tf.sequential();

  //create 'hidden' layer
  let hidden=tf.layers.dense({
    inputShape:[2],//input into layer is 2 values
    units:4,//have 2 nodes in this layer
    activation: 'sigmoid'//use sigmoid activation function
  });

  //create output layer
  let output=tf.layers.dense({
    units:1,//have 1 node in this layer
    activation: 'sigmoid'//use sigmoid activation function
  });

  //add layers into the model
  model.add(hidden);
  model.add(output);

  //compile the model
  let opt=tf.train.adam(.5);//this way can set learning rate to what you want
  model.compile({
    optimizer: opt,
    loss: 'meanSquaredError'
  });

  //build input data for the model
  cols=width/res;
  rows=height/res;
  for(let i=0;i<cols;i++){
    for(let j=0;j<rows;j++){
      inputs.push([i/cols,j/rows]);
    }
  }
  //train the model
  setTimeout(trainModel,10);
}

function trainModel(){
  model.fit(trIn,trOut,{
    shuffle:true,
    epochs:10
  }).then((result)=>{
    setTimeout(trainModel,10);
  });
}

function draw(){
  //get output from the model for the data
  let outputs;
  tf.tidy(()=>{//tidy since making a tensor2d
    outputs=model.predict(tf.tensor2d(inputs)).dataSync();
  });

  //draw rectangles using output(1 or 0) to decide fill color(black or white)
  let index=0;
  for(let i=0;i<cols;i++){
    for(let j=0;j<rows;j++){
      let y=outputs[index]*255;
      fill(y);
      rect(i*res,j*res,res,res);
      fill(255-y);
      textAlign(CENTER,CENTER);
      text(nf(outputs[index],1,2),i*res+res/2,j*res+res/2);
      index++;
    }
  }
}
