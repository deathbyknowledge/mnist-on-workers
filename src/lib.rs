pub mod model;

use crate::model::{Model, NDBackend};
use burn::tensor::Tensor;
use worker::*;

#[durable_object]
pub struct MNISTClassifier {
    model: Option<Model<NDBackend>>,
    _state: State,
    env: Env,
}

#[durable_object]
impl DurableObject for MNISTClassifier {
    fn new(state: State, env: Env) -> Self {
        Self {
            model: None,
            _state: state,
            env,
        }
    }

    async fn fetch(&mut self, mut req: Request) -> Result<Response> {
        if self.model.is_none() {
            self.load_model().await?;
        }

        // Expects it to be an array of 28 * 28 floats
        let numbers: Vec<f32> = req.json().await?;

        // Array of probabilities for our lables (0 to 9)
        let result = self.classify(&numbers);

        Response::from_json(&result)
    }
}

impl MNISTClassifier {
    /// Classify the input image [f32; 28*28] and return the array of probabilities.
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

    /// Fetch model weights from R2 and load the model into the DO
    async fn load_model(&mut self) -> Result<()> {
        use burn::{
            module::Module,
            record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        };
        let bucket = self.env.bucket("BUCKET")?;
        let obj = bucket
            .get("mnist.bin")
            .execute()
            .await?
            .expect("Couldn't find model weights, did you forget to upload them?");
        let bytes = obj
            .body()
            .expect("Failed to read object body")
            .bytes()
            .await?;
        let model: Model<NDBackend> = Model::new(&Default::default());
        let record = BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(bytes, &Default::default())
            .expect("Failed to decode state");

        self.model = Some(model.load_record(record));
        Ok(())
    }
}

#[event(fetch)]
async fn fetch(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    console_error_panic_hook::set_once();

    // Durable Objects get geographically pinned, so we'll instantiate
    // them by the source continent of the request.
    if req.method() == Method::Post && req.path().contains("/classify") {
        // Read request's continent
        let continent = req
            .cf()
            .expect("Failed to read CF request info")
            .continent()
            .expect("Failed to read CF Continent");

        let model_runner = env
            .durable_object("CLASSIFIER")?
            .id_from_name(&continent)?
            .get_stub()?;
        model_runner.fetch_with_request(req).await
    } else {
        return env.assets("ASSETS")?.fetch_request(req).await;
    }
}
