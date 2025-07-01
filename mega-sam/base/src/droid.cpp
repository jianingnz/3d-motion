#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> projective_transform_cuda(torch::Tensor poses,
                                                     torch::Tensor disps,
                                                     torch::Tensor intrinsics,
                                                     torch::Tensor ii,
                                                     torch::Tensor jj);

torch::Tensor depth_filter_cuda(torch::Tensor poses, torch::Tensor disps,
                                torch::Tensor intrinsics, torch::Tensor ix,
                                torch::Tensor thresh, const int model_id);

torch::Tensor frame_distance_cuda(torch::Tensor poses, torch::Tensor disps,
                                  torch::Tensor intrinsics, torch::Tensor ii,
                                  torch::Tensor jj, const float beta,
                                  const int model_id);

std::vector<torch::Tensor> projmap_cuda(torch::Tensor poses,
                                        torch::Tensor disps,
                                        torch::Tensor intrinsics,
                                        torch::Tensor ii, torch::Tensor jj);

torch::Tensor iproj_cuda(torch::Tensor poses, torch::Tensor disps,
                         torch::Tensor intrinsics, const int model_id);

std::vector<torch::Tensor> ba_cuda(
    torch::Tensor poses, torch::Tensor disps, torch::Tensor intrinsics,
    torch::Tensor disps_sens, torch::Tensor targets, torch::Tensor weights,
    torch::Tensor eta, torch::Tensor ii, torch::Tensor jj, torch::Tensor Calib_,
    torch::Tensor CalibPose_, torch::Tensor CalibDepth_, torch::Tensor q_,
    torch::Tensor Hs_, torch::Tensor vs_, torch::Tensor Eii_,
    torch::Tensor Eij_, torch::Tensor Cii_, torch::Tensor wi_, const int t0,
    const int t1, const int iterations, const int model_id, const float lm,
    const float ep, const bool motion_only, const bool optimize_intrinsics,
    const float alpha);

std::vector<torch::Tensor> corr_index_cuda_forward(torch::Tensor volume,
                                                   torch::Tensor coords,
                                                   int radius);

std::vector<torch::Tensor> corr_index_cuda_backward(torch::Tensor volume,
                                                    torch::Tensor coords,
                                                    torch::Tensor corr_grad,
                                                    int radius);

std::vector<torch::Tensor> altcorr_cuda_forward(torch::Tensor fmap1,
                                                torch::Tensor fmap2,
                                                torch::Tensor coords,
                                                int radius);

std::vector<torch::Tensor> altcorr_cuda_backward(torch::Tensor fmap1,
                                                 torch::Tensor fmap2,
                                                 torch::Tensor coords,
                                                 torch::Tensor corr_grad,
                                                 int radius);

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> ba(
    torch::Tensor poses, torch::Tensor disps, torch::Tensor intrinsics,
    torch::Tensor disps_sens, torch::Tensor targets, torch::Tensor weights,
    torch::Tensor eta, torch::Tensor ii, torch::Tensor jj, torch::Tensor Calib,
    torch::Tensor CalibPose, torch::Tensor CalibDepth, torch::Tensor q_vec,
    torch::Tensor Hs, torch::Tensor vs, torch::Tensor Eii, torch::Tensor Eij,
    torch::Tensor Cii, torch::Tensor wi, const int t0, const int t1,
    const int iterations, const int model_id, const float lm, const float ep,
    const bool motion_only, const bool optimize_intrinsics, const float alpha) {
  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(disps_sens);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return ba_cuda(poses, disps, intrinsics, disps_sens, targets, weights, eta,
                 ii, jj, Calib, CalibPose, CalibDepth, q_vec, Hs, vs, Eii, Eij,
                 Cii, wi, t0, t1, iterations, model_id, lm, ep, motion_only,
                 optimize_intrinsics, alpha);
}

torch::Tensor frame_distance(torch::Tensor poses, torch::Tensor disps,
                             torch::Tensor intrinsics, torch::Tensor ii,
                             torch::Tensor jj, const float beta,
                             const int model_id) {
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return frame_distance_cuda(poses, disps, intrinsics, ii, jj, beta, model_id);
}

std::vector<torch::Tensor> projmap(torch::Tensor poses, torch::Tensor disps,
                                   torch::Tensor intrinsics, torch::Tensor ii,
                                   torch::Tensor jj) {
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return projmap_cuda(poses, disps, intrinsics, ii, jj);
}

torch::Tensor iproj(torch::Tensor poses, torch::Tensor disps,
                    torch::Tensor intrinsics, const int model_id) {
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);

  return iproj_cuda(poses, disps, intrinsics, model_id);
}

// c++ python binding
std::vector<torch::Tensor> corr_index_forward(torch::Tensor volume,
                                              torch::Tensor coords,
                                              int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);

  return corr_index_cuda_forward(volume, coords, radius);
}

std::vector<torch::Tensor> corr_index_backward(torch::Tensor volume,
                                               torch::Tensor coords,
                                               torch::Tensor corr_grad,
                                               int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  auto volume_grad =
      corr_index_cuda_backward(volume, coords, corr_grad, radius);
  return {volume_grad};
}

std::vector<torch::Tensor> altcorr_forward(torch::Tensor fmap1,
                                           torch::Tensor fmap2,
                                           torch::Tensor coords, int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);

  return altcorr_cuda_forward(fmap1, fmap2, coords, radius);
}

std::vector<torch::Tensor> altcorr_backward(torch::Tensor fmap1,
                                            torch::Tensor fmap2,
                                            torch::Tensor coords,
                                            torch::Tensor corr_grad,
                                            int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  return altcorr_cuda_backward(fmap1, fmap2, coords, corr_grad, radius);
}

torch::Tensor depth_filter(torch::Tensor poses, torch::Tensor disps,
                           torch::Tensor intrinsics, torch::Tensor ix,
                           torch::Tensor thresh, const int model_id) {
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ix);
  CHECK_INPUT(thresh);

  return depth_filter_cuda(poses, disps, intrinsics, ix, thresh, model_id);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // bundle adjustment kernels
  m.def("ba", &ba, "bundle adjustment");
  m.def("frame_distance", &frame_distance, "frame_distance");
  m.def("projmap", &projmap, "projmap");
  m.def("depth_filter", &depth_filter, "depth_filter");
  m.def("iproj", &iproj, "back projection");

  // correlation volume kernels
  m.def("altcorr_forward", &altcorr_forward, "ALTCORR forward");
  m.def("altcorr_backward", &altcorr_backward, "ALTCORR backward");
  m.def("corr_index_forward", &corr_index_forward, "INDEX forward");
  m.def("corr_index_backward", &corr_index_backward, "INDEX backward");
}