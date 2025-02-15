pub mod model;

use crate::model::{Model, NDBackend};
use burn::tensor::Tensor;
use worker::*;

#[durable_object]
pub struct ModelRunner {
    model: Option<Model<NDBackend>>,
    state: State,
    env: Env,
}

#[durable_object]
impl DurableObject for ModelRunner {
    fn new(state: State, env: Env) -> Self {
        Self {
            model: None,
            state: state,
            env,
        }
    }

    async fn fetch(&mut self, mut req: Request) -> Result<Response> {
        if self.model.is_none() {
            self.load_model().await?;
        }

        let numbers: Vec<f32> = req.json().await?;
        console_log!("{:?}", numbers);

        let result = self.classify(&numbers);
        console_log!("{:?}", result);

        Response::from_json(&result)
    }
}

impl ModelRunner {
    fn classify(&mut self, input: &[f32]) -> Vec<f32> {
        let device = Default::default();
        let input = Tensor::<NDBackend, 1>::from_floats(input, &device).reshape([1, 28, 28]);
        let input = ((input / 255) - 0.1307) / 0.3081;
        // Normalize input: make between [0,1] and make the mean=0 and std=1
        // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
        // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122

        let output = self.model.as_ref().unwrap().forward(input);

        let output = burn::tensor::activation::softmax(output, 1);
        let predictions = output.into_data();
        predictions.to_vec().unwrap()
    }

    async fn load_model(&mut self) -> Result<()> {
        use burn::{
            module::Module,
            record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        };
        console_log!("No model present. Initializing.");
        let bucket = self.env.bucket("BUCKET")?;
        console_log!("Fetching model from R2...");
        let obj = bucket.get("model.bin").execute().await?.expect("msg");
        let bytes = obj.body().unwrap().bytes().await?;
        console_log!("Model weights acquired. Loading into model...");
        let model: Model<NDBackend> = Model::new(&Default::default());
        let record = BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(bytes, &Default::default())
            .expect("Failed to decode state");

        console_log!("Successfully loaded model.");
        self.model = Some(model.load_record(record));
        Ok(())
    }
}

#[event(fetch)]
async fn fetch(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    console_error_panic_hook::set_once();

    if req.path().contains("/classify") {
        let model_runner = env
            .durable_object("MODEL")?
            .id_from_name("MY_MODEL_RUNNER")?
            .get_stub()?;
        model_runner.fetch_with_request(req).await
    } else {
        return env.assets("ASSETS")?.fetch_request(req).await;
    }
}
