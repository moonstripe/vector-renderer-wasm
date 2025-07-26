use std::process::Command;

fn main() {
    let _ = Command::new("cp")
        .args(["-v", "./src/main.rs", "./src/lib.rs"])
        .output()
        .expect("Could not copy ./src/main.rs to ./src/lib.rs");
}
