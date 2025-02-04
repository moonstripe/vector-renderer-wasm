use core::f32;

use bevy::{
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    reflect::TypePath,
    render::{
        mesh::{MeshVertexBufferLayoutRef, PrimitiveTopology},
        render_asset::RenderAssetUsages,
        render_resource::{
            AsBindGroup, PolygonMode, RenderPipelineDescriptor, ShaderRef,
            SpecializedMeshPipelineError,
        },
    },
};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use bevy_mod_picking::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use once_cell::sync::Lazy;
use rand::Rng;
use serde::Deserialize;
use std::sync::Mutex;
use wasm_bindgen::prelude::*;

const AXIS_LENGTH: f32 = 500.;

// TODO: Make sure that [f32; 3] is replaced with a copyable struct with an id to delete.
static EMBEDDING_QUEUE: Lazy<Mutex<Vec<[f32; 3]>>> = Lazy::new(|| Mutex::new(Vec::new()));

#[derive(Resource)]
struct EmbeddingQueueResource;

fn process_new_embeddings(mut renderer_state: ResMut<RendererState>) {
    info!("process_new_embeddings");
    let start = instant::now();
    match EMBEDDING_QUEUE.try_lock() {
        Ok(mut queue) => {
            for embedding in queue.drain(..) {
                let text = TextEmbedding {
                    embedding,
                    drawn: false,
                };
                if let Some(texts) = &mut renderer_state.texts {
                    texts.push(text);
                } else {
                    renderer_state.texts = Some(vec![text]);
                }
            }
            let end = instant::now();

            info!("process_new_embeddings: {:#?}", end - start);
        }
        Err(e) => {
            warn!("could not get lock... {:#?}", e);
            let end = instant::now();

            info!("process_new_embeddings: {:#?}", end - start);
        }
    };
}

#[wasm_bindgen]
pub fn add_embedding(x: f32, y: f32, z: f32) {
    let start = instant::Instant::now();
    let mut queue = EMBEDDING_QUEUE.lock().unwrap();

    queue.push([x, y, z]);
    let end = instant::Instant::now();
    info!("Startup Lazy Queue: {:#?}", end - start);
}

#[derive(Component, Debug, Clone, Copy, Deserialize)]
struct TextEmbedding {
    embedding: [f32; 3],
    drawn: bool,
}

#[derive(Asset, TypePath, Default, AsBindGroup, Debug, Clone)]
struct LineMaterial {
    #[uniform(0)]
    color: LinearRgba,
}

impl Material for LineMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/line_material.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // This is the important part to tell bevy to render this material as a line between vertices
        descriptor.primitive.polygon_mode = PolygonMode::Fill;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct Axis {
    line: [Vec3; 2],
    color: LinearRgba,
}

#[derive(Debug, Clone, Resource)]
struct RendererState {
    texts: Option<Vec<TextEmbedding>>,
}

impl From<Axis> for Mesh {
    fn from(axis: Axis) -> Self {
        let vertices = axis.line.to_vec();

        Mesh::new(
            // This tells wgpu that the positions are list of lines
            // where every pair is a start and end point
            PrimitiveTopology::LineList,
            RenderAssetUsages::RENDER_WORLD,
        )
        // Add the vertices positions as an attribute
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    }
}

fn startup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
            ..default()
        },
        PanOrbitCamera {
            zoom_upper_limit: Some(5.0),
            zoom_lower_limit: Some(1.0),
            ..default()
        },
    ));

    // Define axes
    let origin = Vec3::new(0., 0., 0.);
    let x_axis_end = Vec3::X * AXIS_LENGTH;
    let y_axis_end = Vec3::Y * AXIS_LENGTH;
    let z_axis_end = Vec3::Z * AXIS_LENGTH;

    let x_axis = Axis {
        line: [origin, x_axis_end],
        color: LinearRgba::RED,
    };
    let y_axis = Axis {
        line: [origin, y_axis_end],
        color: LinearRgba::GREEN,
    };
    let z_axis = Axis {
        line: [origin, z_axis_end],
        color: LinearRgba::BLUE,
    };

    // Spawn X Axis
    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(x_axis),
        material: line_materials.add(LineMaterial {
            color: x_axis.color,
        }),
        ..default()
    });

    // Spawn Y Axis
    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(y_axis),
        material: line_materials.add(LineMaterial {
            color: y_axis.color,
        }),
        ..default()
    });

    // Spawn Z Axis
    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(z_axis),
        material: line_materials.add(LineMaterial {
            color: z_axis.color,
        }),
        ..default()
    });
}

fn update(mut contexts: EguiContexts, mut renderer_state: ResMut<RendererState>) {
    egui::Window::new("Options")
        .movable(false)
        .show(contexts.ctx_mut(), |ui| {
            let num_texts = match &mut renderer_state.texts {
                Some(num_texts) => num_texts.len(),
                None => 0,
            };

            ui.label(format!("Rendering {} Text Embeddings.", num_texts));

            if ui.button("Add Random Text Embedding").clicked() {
                info!("Adding Random Text Embedding...");

                let mut rng = rand::thread_rng();
                let x = rng.gen_range(-1.0..1.0);
                let y = rng.gen_range(-1.0..1.0);
                let z = rng.gen_range(-1.0..1.0);

                let text = TextEmbedding {
                    embedding: [x, y, z],
                    drawn: false,
                };

                let mut queue = EMBEDDING_QUEUE.lock().unwrap();

                queue.push(text.embedding)
            }
        });
}

fn draw_new_embeddings(
    mut commands: Commands,
    mut renderer_state: ResMut<RendererState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if let Some(texts) = &mut renderer_state.texts {
        for text in texts {
            if !text.drawn {
                commands
                    .spawn(PbrBundle {
                        mesh: meshes.add(Sphere::new(0.01).mesh()),
                        material: materials.add(StandardMaterial {
                            base_color: Color::srgba(
                                text.embedding[0],
                                text.embedding[1],
                                text.embedding[2],
                                1.0,
                            ),
                            emissive: LinearRgba::new(
                                text.embedding[0],
                                text.embedding[1],
                                text.embedding[2],
                                1.0,
                            ),
                            alpha_mode: AlphaMode::Blend,
                            ..default()
                        }),
                        transform: Transform::from_xyz(
                            text.embedding[0],
                            text.embedding[1],
                            text.embedding[2],
                        ),
                        ..default()
                    })
                    .insert(*text)
                    .insert(On::<Pointer<Click>>::target_component_mut::<Transform>(
                        |_pos, transform| {
                            if transform.scale == Vec3::ONE {
                                transform.scale *= 3.;
                            } else {
                                transform.scale = Vec3::ONE
                            }
                        },
                    ));

                text.drawn = true;
            };
        }
    }
}

fn blur_embeddings(
    camera_query: Query<&Transform, With<Camera>>,
    mut emb_query: Query<(&Transform, &Handle<StandardMaterial>), With<TextEmbedding>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let camera_transform = camera_query.single();

    for (&emb_transform, emb_material) in emb_query.iter_mut() {
        let depth = (camera_transform.translation - emb_transform.translation).length();

        // Steeper drop-off curve
        let opacity = if depth >= 5.0 {
            0.1
        } else {
            1.0 - (depth / 5.0) * 0.9
        };

        // Update the material's opacity
        if let Some(material) = materials.get_mut(emb_material) {
            info!("updating material. a: {}, mat: {:#?}", opacity, material);
            material.base_color.set_alpha(opacity);
            material.emissive.set_alpha(opacity);
        }
    }
}

#[wasm_bindgen(start)]
pub fn run() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Vector Renderer Test".to_string(),
                    canvas: Some("#vector-canvas".into()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            MaterialPlugin::<LineMaterial>::default(),
            PanOrbitCameraPlugin,
            EguiPlugin,
        ))
        .add_plugins(DefaultPickingPlugins)
        .insert_resource(RendererState { texts: None })
        .insert_resource(EmbeddingQueueResource)
        .add_systems(Startup, startup)
        .add_systems(PreUpdate, process_new_embeddings)
        .add_systems(Update, (update, draw_new_embeddings))
        .add_systems(PostUpdate, blur_embeddings)
        .run();
}
