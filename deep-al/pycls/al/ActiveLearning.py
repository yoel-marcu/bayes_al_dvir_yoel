# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

from .Sampling import Sampling, CoreSetMIPSampling, AdversarySampler
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

class ActiveLearning:
    """
    Implements standard active learning methods.
    """

    def __init__(self, dataObj, cfg, train_labels=None, lset=None):
        self.dataObj = dataObj
        self.sampler = Sampling(dataObj=dataObj,cfg=cfg)
        self.cfg = cfg
        self.sampling_fn = self.choose_sampling_function(train_labels, lset)

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None):
        """
        Sample from uSet using cfg.ACTIVE_LEARNING.SAMPLING_FN.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]

        supportingModels: List of models which are used for sampling process.

        OUTPUT
        -------
        Returns activeSet, uSet
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
        .format(len(uSet), self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)

        if self.sampling_fn is not None:
            logger.info(f"Using {self.cfg.ACTIVE_LEARNING.SAMPLING_FN} sampling function for active learning.")
            return self.sampling_fn.select_samples(lSet, uSet)


        if self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":

            activeSet, uSet = self.sampler.random(uSet=uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.uncertainty(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "entropy":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.entropy(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "margin":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.margin(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "coreset":
            waslatent = clf_model.penultimate_active
            wastrain = clf_model.training
            clf_model.penultimate_active = True
            # if self.cfg.TRAIN.DATASET == "IMAGENET":
            #     clf_model.cuda(0)
            clf_model.eval()
            coreSetSampler = CoreSetMIPSampling(cfg=self.cfg, dataObj=self.dataObj)
            activeSet, uSet = coreSetSampler.query(lSet=lSet, uSet=uSet, clf_model=clf_model, dataset=trainDataset)
            
            clf_model.penultimate_active = waslatent
            clf_model.train(wastrain)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("typiclust"):
            from .typiclust import TypiClust
            is_scan = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.endswith('dc')
            tpc = TypiClust(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, is_scan=is_scan)
            activeSet, uSet = tpc.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["prob_cover", 'probcover']:
            from .prob_cover import ProbCover
            probcov = ProbCover(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
            activeSet, uSet = probcov.select_samples()
            # probcov.plot_tsne()
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["misp"]:
            from .MISP import  MISP

            misp = MISP(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
            activeSet, uSet = misp.select_samples()
            # misp.plot_tsne()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["mispc"]:
            from .MISP_probcover import MISPC

            misp = MISPC(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
            activeSet, uSet = misp.select_samples()
            # misp.plot_tsne()
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["misp_plus"]:
            from .MISP_plus import MISP_PLUS

            mispp = MISP_PLUS(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, train_data=trainDataset,
                        delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
            activeSet, uSet = mispp.select_samples()
            # misp.plot_tsne()
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["maxherding", "max_herding"]:
            from .maxherding import MaxHerding
            delta = self.cfg.ACTIVE_LEARNING.INITIAL_DELTA
            maxherding = MaxHerding(self.cfg, lSet, uSet, self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, delta=delta)
            activeSet, uSet = maxherding.select_samples()
            # maxherding.plot_tsne()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["dcom"]:
            from .DCoM import DCoM
            dcom = DCoM(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        max_delta=self.cfg.ACTIVE_LEARNING.MAX_DELTA,
                        lSet_deltas=self.cfg.ACTIVE_LEARNING.DELTA_LST)
            activeSet, uSet = dcom.select_samples(clf_model, trainDataset, self.dataObj)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "dbal" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "DBAL":
            activeSet, uSet = self.sampler.dbal(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, \
                uSet=uSet, clf_model=clf_model,dataset=trainDataset)
            
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "bald" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "BALD":
            activeSet, uSet = self.sampler.bald(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_model=clf_model, dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "ensemble_var_R":
            activeSet, uSet = self.sampler.ensemble_var_R(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_models=supportingModels, dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "vaal":
            adv_sampler = AdversarySampler(cfg=self.cfg, dataObj=self.dataObj)

            # Train VAE and discriminator first
            vae, disc, uSet_loader = adv_sampler.vaal_perform_training(lSet=lSet, uSet=uSet, dataset=trainDataset)

            # Do active sampling
            activeSet, uSet = adv_sampler.sample_for_labeling(vae=vae, discriminator=disc, \
                                unlabeled_dataloader=uSet_loader, uSet=uSet)
        else:
            print(f"{self.cfg.ACTIVE_LEARNING.SAMPLING_FN} is either not implemented or there is some spelling mistake.")
            raise NotImplementedError

        return activeSet, uSet
        

    def choose_sampling_function(self, train_labels=None, lset=None):
        if self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["prob_cover", 'probcover']:
            from .prob_cover import ProbCover
            probcov = ProbCover(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
            return probcov
            activeSet, uSet = probcov.select_samples(lSet, uSet)
            # probcov.plot_tsne()
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["misp"]:
            from .MISP import  MISP

            misp = MISP(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
            return misp
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["max_misp"]:
            from .MAX_MISP import  MAX_MISP

            max_misp = MAX_MISP(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
            return max_misp

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["all_misp"]:
            from .ALL_MISP import  ALL_MISP

            all_misp = ALL_MISP(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        train_labels= train_labels, delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA, lset=lset)
            return all_misp

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["bayes_misp"]:
            from .BAYES_MISP import BAYES_MISP

            bayes_misp = BAYES_MISP(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        train_labels= train_labels, delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA, lset=lset)
            return bayes_misp

