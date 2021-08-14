from lxml import etree, objectify
import os.path as osp
import numpy as np
import os
import mmcv

def create_xml(
                filename,
                result,
                outdir,
                img,
                class_names,
                score_thr=0.5
            ):
    """Create the voc format XML for the given results

    Args:
        filename (str): filename
        result (np.array(N*5)): bboxes and score
        outdir (str): output_dir
        img (ndarray): array of img
        class_names (str): class names
        score_thr (float, optional): threshold. Defaults to 0.5.
    """ 
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    gen_xml(filename, img, bboxes, labels, class_names, outdir)
    
    pass

def gen_xml(filename, img, bboxes, labels, class_names, result_pth):
    """Create and write xml

    Args:
        filename (str): filename
        img (ndarray): array of img
        bboxes (ndarray): results
        labels (ndarray): label
        class_names (ndarray[str]): class names
        result_pth (str): path
    """    
    img = mmcv.imread(img)
    E = objectify.ElementMaker(annotate=False)
    assert len(img.shape) == 3
    anno_tree = E.annotation(
        E.folder('Annotations'),
        E.filename(filename),
        E.path(osp.join('../JPEGImages', filename+'.jpg')),
        E.source(
            E.database('Unknown')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented('0')
    )
    E2 = objectify.ElementMaker(annotate=False)
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        anno_tree2 = E2.object(
            E.name(label_text),
            E.pose('Unspecified'),
            E.truncated('0'),
            E.difficult('0'),
            E.bndbox(
                E.xmin(bbox_int[0]),
                E.ymin(bbox_int[1]),
                E.xmax(bbox_int[2]),
                E.ymax(bbox_int[3])
            )
        )
        anno_tree.append(anno_tree2)
    os.makedirs(osp.join(result_pth, 'Annotations'), exist_ok=True)
    etree.ElementTree(anno_tree).write(osp.join(result_pth, 'Annotations', filename+'.xml')
                                        , pretty_print=True, encoding="UTF-8")