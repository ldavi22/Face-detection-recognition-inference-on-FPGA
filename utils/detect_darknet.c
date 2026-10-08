#include "darknet.h"
#include "detect.h"
#include <stdlib.h>
static network *net = NULL;

static int dn_init(const char *cfg, const char *weights) {
  net = load_network((char *)cfg, (char *)weights, 0);
  if (net == NULL)
    return 1;
  set_batch_network(net, 1);
  return 0;
}

static void dn_run(const float *input, yolo_output *out) {
  network_predict(net, (float *)input);
  out->small = net->layers[16].output;
  out->large = net->layers[23].output;
}

static void dn_cleanup() {
  free_network(net);
  net = NULL;
}

const detector_api detector_darknet = {
    .name = "darknet", .init = dn_init, .run = dn_run, .cleanup = dn_cleanup};
